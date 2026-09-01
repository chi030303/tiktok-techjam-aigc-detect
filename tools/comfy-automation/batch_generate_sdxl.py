#!/usr/bin/env python3
"""Resumable, bounded-queue SDXL batch generator for ComfyUI on Vast."""
from __future__ import annotations

import argparse, copy, datetime as dt, hashlib, json, logging, os, re, signal
import sqlite3, sys, time, urllib.error, urllib.parse, urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MODEL = "sdxl"
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SIZE_RE = re.compile(r"^(\d+)[xX](\d+)$")
ASPECT_SIZES = {"1:1": (1024, 1024), "4:5": (896, 1120), "9:16": (768, 1344),
                "3:2": (1216, 832), "16:9": (1344, 768)}
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, lowres, blurry, out of focus, jpeg artifacts, text, watermark, "
    "logo, signature, malformed anatomy, deformed, disfigured, extra limbs, missing limbs, "
    "fused limbs, extra arms, extra legs, malformed hands, poorly drawn hands, extra fingers, "
    "missing fingers, fused fingers, duplicated fingers, broken fingers, distorted face, duplicated person"
)
MODEL_DIRS = {"CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
              "UNETLoader": ("unet_name", "diffusion_models"),
              "CLIPLoader": ("clip_name", "text_encoders"), "VAELoader": ("vae_name", "vae")}


class BatchError(RuntimeError): kind = "internal"
class ValidationError(BatchError): kind = "validation"
class ExecutionError(BatchError): kind = "execution"
class OutputMissingError(BatchError): kind = "missing_output"


class ComfyHTTPError(BatchError):
    def __init__(self, status: int, body: str):
        self.status, self.body = status, body
        self.kind = classify_error(body, "http_400" if status == 400 else "http")
        super().__init__(f"ComfyUI HTTP {status}: {body[:1500]}")


def now() -> str: return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def classify_error(message: str, default: str = "execution") -> str:
    text = message.lower()
    if any(x in text for x in ("out of memory", "cuda oom", "cuda error: out")): return "oom"
    if any(x in text for x in ("prompt_outputs_failed_validation", "invalid prompt", "required input")):
        return "node_validation"
    return default


def stable_seed(sample_id: str, size_key: str, salt: str) -> int:
    raw = hashlib.sha256(f"{salt}\0{MODEL}\0{sample_id}\0{size_key}".encode()).digest()
    return int.from_bytes(raw[:8], "big") & ((1 << 63) - 1)


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle: return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc: raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc


def validate_prompts(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list) or not data: raise ValidationError("Prompt file must be a non-empty JSON array")
    seen, result = set(), []
    for index, item in enumerate(data):
        if not isinstance(item, dict): raise ValidationError(f"Prompt #{index + 1} is not an object")
        sample_id = item.get("prompt_id") or item.get("sample_id") or item.get("id")
        text = item.get("rendered_prompt") or item.get("prompt") or item.get("text")
        if not isinstance(sample_id, str) or not sample_id.strip(): raise ValidationError(f"Prompt #{index + 1} has no stable ID")
        sample_id = sample_id.strip()
        if not SAFE_NAME.fullmatch(sample_id) or sample_id in {".", ".."}: raise ValidationError(f"Unsafe prompt ID: {sample_id!r}")
        if sample_id in seen: raise ValidationError(f"Duplicate prompt ID: {sample_id}")
        if not isinstance(text, str) or not text.strip(): raise ValidationError(f"Prompt {sample_id} has empty text")
        seen.add(sample_id); normalized = dict(item); normalized.update(sample_id=sample_id, prompt_text=text.strip()); result.append(normalized)
    return result


def unsafe_path_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str) and (value.startswith(("/Users/", "/home/")) or ":\\" in value): yield value
    elif isinstance(value, dict):
        for child in value.values(): yield from unsafe_path_strings(child)
    elif isinstance(value, list):
        for child in value: yield from unsafe_path_strings(child)


def linked_node(value: Any) -> str | None:
    return str(value[0]) if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) else None


@dataclass(frozen=True)
class Job:
    sample_id: str; prompt: str; width: int; height: int; size_key: str; seed: int


@dataclass(frozen=True)
class WorkflowSpec:
    template: dict[str, Any]
    positive_targets: tuple[tuple[str, str], ...]
    negative_targets: tuple[tuple[str, str], ...]
    seed_targets: tuple[tuple[str, str], ...]
    save_node: str
    dimension_node: str
    sampler_nodes: tuple[str, ...]
    base_sampler: str
    refiner_sampler: str
    decode_node: str
    base_loader: str
    class_types: frozenset[str]

    @classmethod
    def inspect(cls, path: Path) -> "WorkflowSpec":
        workflow = load_json(path)
        if not isinstance(workflow, dict) or not workflow: raise ValidationError("Workflow must be a non-empty API node map")
        for node_id, node in workflow.items():
            if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
                raise ValidationError(f"Node {node_id!r} is not ComfyUI API format")
        bad = list(unsafe_path_strings(workflow))
        if bad: raise ValidationError(f"Workflow contains local absolute paths: {bad}")
        samplers = [str(k) for k, v in workflow.items() if v["class_type"].startswith("KSampler")]
        if not samplers: raise ValidationError("Workflow has no KSampler")
        positive_ids = {linked_node(workflow[n]["inputs"].get("positive")) for n in samplers} - {None}
        negative_ids = {linked_node(workflow[n]["inputs"].get("negative")) for n in samplers} - {None}
        def text_targets(ids: set[str]) -> tuple[tuple[str, str], ...]:
            return tuple((n, "text") for n in sorted(ids)
                         if n in workflow and workflow[n]["class_type"] == "CLIPTextEncode"
                         and isinstance(workflow[n]["inputs"].get("text"), str))
        positives, negatives = text_targets(positive_ids), text_targets(negative_ids)
        seeds = tuple((n, f) for n in samplers for f in ("noise_seed", "seed")
                      if isinstance(workflow[n]["inputs"].get(f), int))
        saves = [str(k) for k, v in workflow.items() if v["class_type"] == "SaveImage"]
        dimensions = [str(k) for k, v in workflow.items()
                      if isinstance(v["inputs"].get("width"), int) and isinstance(v["inputs"].get("height"), int)]
        if not positives: raise ValidationError("No positive CLIPTextEncode connected to samplers")
        if not negatives: raise ValidationError("No negative CLIPTextEncode connected to samplers")
        if not seeds: raise ValidationError("No sampler seed inputs found")
        if len(saves) != 1: raise ValidationError(f"Expected one SaveImage, found {saves}")
        if len(dimensions) != 1: raise ValidationError(f"Expected one width/height source, found {dimensions}")
        base_samplers = [n for n in samplers if workflow[n]["inputs"].get("add_noise") != "disable"]
        refiner_samplers = [n for n in samplers if workflow[n]["inputs"].get("add_noise") == "disable"]
        decode_node = linked_node(workflow[saves[0]]["inputs"].get("images"))
        base_loader = linked_node(workflow[base_samplers[0]]["inputs"].get("model")) if len(base_samplers) == 1 else None
        if len(base_samplers) != 1 or len(refiner_samplers) != 1:
            raise ValidationError(f"Expected one Base and one Refiner sampler, found base={base_samplers}, refiner={refiner_samplers}")
        if not decode_node or decode_node not in workflow or workflow[decode_node]["class_type"] != "VAEDecode":
            raise ValidationError("SaveImage is not connected to an identifiable VAEDecode node")
        if not base_loader or base_loader not in workflow or workflow[base_loader]["class_type"] != "CheckpointLoaderSimple":
            raise ValidationError("Base sampler model is not connected to CheckpointLoaderSimple")
        return cls(workflow, positives, negatives, seeds, saves[0], dimensions[0], tuple(samplers),
                   base_samplers[0], refiner_samplers[0], decode_node, base_loader,
                   frozenset(v["class_type"] for v in workflow.values()))

    def render(self, job: Job, run_name: str | None, negative: str, preset: str,
               disable_refiner: bool = False) -> dict[str, Any]:
        workflow = copy.deepcopy(self.template)
        for n, f in self.positive_targets: workflow[n]["inputs"][f] = job.prompt
        for n, f in self.negative_targets: workflow[n]["inputs"][f] = negative
        for n, f in self.seed_targets: workflow[n]["inputs"][f] = job.seed
        workflow[self.dimension_node]["inputs"].update(width=job.width, height=job.height)
        if preset == "quality":
            for n in self.sampler_nodes:
                inputs = workflow[n]["inputs"]
                inputs.update(steps=30, cfg=6.5, sampler_name="dpmpp_2m", scheduler="karras")
                if inputs.get("add_noise") == "disable": inputs.update(start_at_step=24, end_at_step=10000)
                else: inputs.update(start_at_step=0, end_at_step=24)
        if disable_refiner:
            base_inputs = workflow[self.base_sampler]["inputs"]
            base_inputs["start_at_step"] = 0
            base_inputs["end_at_step"] = 10000
            base_inputs["return_with_leftover_noise"] = "disable"
            workflow[self.decode_node]["inputs"]["samples"] = [self.base_sampler, 0]
            workflow[self.decode_node]["inputs"]["vae"] = [self.base_loader, 2]
        prefix = f"{MODEL}/{job.sample_id}_{job.size_key}_{MODEL}"
        if run_name: prefix = f"{run_name}/{prefix}"
        workflow[self.save_node]["inputs"]["filename_prefix"] = prefix
        return workflow

    def summary(self, preset: str, negative: str, disable_refiner: bool = False) -> dict[str, Any]:
        files = []
        for node in self.template.values():
            mapping = MODEL_DIRS.get(node["class_type"])
            if mapping and mapping[0] in node["inputs"]: files.append({"class_type": node["class_type"], "file": node["inputs"][mapping[0]]})
        settings = [{"node": n, **{k: self.template[n]["inputs"].get(k) for k in
                     ("add_noise", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step")}}
                    for n in self.sampler_nodes]
        return {"positive_prompt_targets": [f"{n}.{f}" for n, f in self.positive_targets],
                "negative_prompt_targets": [f"{n}.{f}" for n, f in self.negative_targets],
                "seed_targets": [f"{n}.{f}" for n, f in self.seed_targets],
                "dimension_node": self.dimension_node, "save_node": self.save_node,
                "base_sampler": self.base_sampler, "refiner_sampler": self.refiner_sampler,
                "decode_node": self.decode_node, "base_loader": self.base_loader,
                "model_files": files, "workflow_sampler_settings": settings,
                "effective_sampling_preset": preset, "refiner_enabled": not disable_refiner,
                "negative_prompt": negative,
                "class_types": sorted(self.class_types)}


def parse_size(value: str) -> tuple[int, int]:
    match = SIZE_RE.fullmatch(value.strip())
    if not match: raise ValidationError(f"Invalid size {value!r}; expected WIDTHxHEIGHT")
    width, height = map(int, match.groups())
    if any(x < 64 or x % 8 for x in (width, height)): raise ValidationError(f"Size {value!r} must be >=64 and divisible by 8")
    if width * height > 2_100_000: raise ValidationError(f"Size {value!r} exceeds 2.1MP safety limit")
    return width, height


def prompt_size(prompt: dict[str, Any]) -> tuple[int, int]:
    raw = prompt.get("aspect_ratio")
    match = re.search(r"\b(1:1|4:5|9:16|3:2|16:9)\b", raw) if isinstance(raw, str) else None
    if not match: raise ValidationError(f"Unsupported aspect_ratio {raw!r} for {prompt['sample_id']}")
    return ASPECT_SIZES[match.group(1)]


def make_jobs(prompts: list[dict[str, Any]], mode: str, sizes: list[tuple[int, int]], salt: str) -> list[Job]:
    result = []
    for p in prompts:
        selected = [prompt_size(p)] if mode == "prompt" else sizes
        for width, height in selected:
            key = f"{width}x{height}"
            result.append(Job(p["sample_id"], p["prompt_text"], width, height, key, stable_seed(p["sample_id"], key, salt)))
    return result


class ComfyClient:
    def __init__(self, base_url: str, timeout: float): self.base_url, self.timeout = base_url.rstrip("/"), timeout
    def request(self, method: str, path: str, payload: Any = None) -> Any:
        request = urllib.request.Request(self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(), method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response: raw = response.read()
        except urllib.error.HTTPError as exc: raise ComfyHTTPError(exc.code, exc.read().decode("utf-8", "replace")) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = BatchError(f"Cannot reach ComfyUI at {self.base_url}: {exc}"); error.kind = "transport"; raise error from exc
        if not raw: return {}
        try: return json.loads(raw)
        except json.JSONDecodeError as exc: raise BatchError(f"Invalid JSON from {path}: {raw[:500]!r}") from exc
    def get(self, path: str) -> Any: return self.request("GET", path)
    def submit(self, workflow: dict[str, Any], client_id: str) -> str:
        response = self.request("POST", "/prompt", {"prompt": workflow, "client_id": client_id}); prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id: raise ValidationError(f"/prompt did not return prompt_id: {response}")
        return prompt_id
    def history(self, prompt_id: str) -> dict[str, Any] | None:
        response = self.get("/history/" + urllib.parse.quote(prompt_id, safe="")); entry = response.get(prompt_id) if isinstance(response, dict) else None
        return entry if isinstance(entry, dict) else None


class StateDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.connection = sqlite3.connect(path); self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL"); self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (sample_id TEXT NOT NULL, model TEXT NOT NULL, size_key TEXT NOT NULL,
          width INTEGER NOT NULL, height INTEGER NOT NULL, prompt TEXT NOT NULL, negative_prompt TEXT NOT NULL,
          seed INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', prompt_id TEXT, attempt INTEGER NOT NULL DEFAULT 0,
          output_files TEXT NOT NULL DEFAULT '[]', error_type TEXT, error_message TEXT, created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL, PRIMARY KEY(sample_id,model,size_key));
        CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
          model TEXT NOT NULL, size_key TEXT NOT NULL, attempt INTEGER NOT NULL, prompt_id TEXT, status TEXT NOT NULL,
          error_type TEXT, error_message TEXT, output_files TEXT NOT NULL DEFAULT '[]', timestamp TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(model,status);""")
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(jobs)")}
        if "generation_config" not in columns:
            with self.connection:
                self.connection.execute("ALTER TABLE jobs ADD COLUMN generation_config TEXT NOT NULL DEFAULT ''")
    def get(self, sample_id: str, size_key: str) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM jobs WHERE sample_id=? AND model=? AND size_key=?", (sample_id, MODEL, size_key)).fetchone()
    def seed_jobs(self, jobs: list[Job], negative: str, generation_config: str) -> None:
        stamp = now()
        with self.connection:
            self.connection.executemany("""INSERT OR IGNORE INTO jobs
              (sample_id,model,size_key,width,height,prompt,negative_prompt,seed,created_at,updated_at,generation_config)
              VALUES(?,?,?,?,?,?,?,?,?,?,?)""", [(j.sample_id, MODEL, j.size_key, j.width, j.height, j.prompt, negative, j.seed, stamp, stamp, generation_config) for j in jobs])
        for j in jobs:
            row = self.get(j.sample_id, j.size_key)
            if not row or (row["prompt"], row["negative_prompt"], row["seed"], row["width"], row["height"], row["generation_config"]) != (j.prompt, negative, j.seed, j.width, j.height, generation_config):
                raise ValidationError(f"State conflict for ({j.sample_id},{j.size_key}); use a new --state-dir")
    def select(self, jobs: list[Job]) -> list[sqlite3.Row]: return [self.get(j.sample_id, j.size_key) for j in jobs]
    def update(self, row: sqlite3.Row, status: str, *, prompt_id: str | None = None, output_files: list[str] | None = None,
               error_type: str | None = None, error_message: str | None = None, increment_attempt=False, clear_prompt_id=False) -> sqlite3.Row:
        fields, values = ["status=?", "updated_at=?", "error_type=?", "error_message=?"], [status, now(), error_type, error_message]
        if clear_prompt_id: fields.append("prompt_id=NULL")
        elif prompt_id is not None: fields.append("prompt_id=?"); values.append(prompt_id)
        if output_files is not None: fields.append("output_files=?"); values.append(json.dumps(output_files, ensure_ascii=False))
        if increment_attempt: fields.append("attempt=attempt+1")
        values += [row["sample_id"], MODEL, row["size_key"]]
        with self.connection: self.connection.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE sample_id=? AND model=? AND size_key=?", values)
        return self.get(row["sample_id"], row["size_key"])
    def event(self, row: sqlite3.Row, status: str, error_type=None, error_message=None, output_files=None) -> None:
        with self.connection: self.connection.execute("""INSERT INTO attempts
          (sample_id,model,size_key,attempt,prompt_id,status,error_type,error_message,output_files,timestamp)
          VALUES(?,?,?,?,?,?,?,?,?,?)""", (row["sample_id"], MODEL, row["size_key"], row["attempt"], row["prompt_id"], status,
          error_type, error_message, json.dumps(output_files or [], ensure_ascii=False), now()))
    def close(self): self.connection.close()


def extract_execution_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status", {})
    return (json.dumps(status.get("messages", []), ensure_ascii=False)[:12000] or "ComfyUI execution error") if isinstance(status, dict) and status.get("status_str") == "error" else None


def extract_outputs(entry: dict[str, Any], save_node: str, output_dir: Path) -> list[str]:
    output = entry.get("outputs", {}).get(save_node, {}); images = output.get("images", []) if isinstance(output, dict) else []
    files, root = [], output_dir.resolve()
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("filename"), str) or image.get("type", "output") != "output": continue
        path = (output_dir / str(image.get("subfolder", "")) / image["filename"]).resolve()
        try: path.relative_to(root)
        except ValueError as exc: raise ValidationError(f"Output outside output directory: {path}") from exc
        files.append(str(path))
    if not files: raise OutputMissingError(f"No images in history for SaveImage {save_node}")
    missing = [p for p in files if not Path(p).is_file() or Path(p).stat().st_size <= 0]
    if missing: raise OutputMissingError(f"History reports missing or empty files: {missing}")
    return files


def output_files_exist(row: sqlite3.Row) -> bool:
    try: files = json.loads(row["output_files"])
    except (json.JSONDecodeError, TypeError): return False
    return bool(files) and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in files)


def contains_value(value: Any, needle: str) -> bool:
    if value == needle: return True
    if isinstance(value, dict): return any(contains_value(v, needle) for v in value.values())
    if isinstance(value, (list, tuple)): return any(contains_value(v, needle) for v in value)
    return False


def queue_depth(queue: Any) -> int:
    if not isinstance(queue, dict): return 0
    return sum(len(queue.get(k, [])) for k in ("queue_running", "queue_pending") if isinstance(queue.get(k, []), list))


def model_files(spec: WorkflowSpec, comfy_dir: Path) -> list[Path]:
    result = []
    for node in spec.template.values():
        mapping = MODEL_DIRS.get(node["class_type"])
        if mapping and isinstance(node["inputs"].get(mapping[0]), str): result.append(comfy_dir / "models" / mapping[1] / node["inputs"][mapping[0]])
    return result


def preflight(client: ComfyClient, spec: WorkflowSpec, comfy_dir: Path, output_dir: Path | None, skip_models: bool) -> Path:
    stats = client.get("/system_stats"); logging.info("system_stats devices=%s", stats.get("devices", "unknown"))
    system = stats.get("system", {}) if isinstance(stats, dict) else {}; reported = system.get("output_directory") if isinstance(system, dict) else None
    reported = reported or (stats.get("output_directory") if isinstance(stats, dict) else None)
    output_dir = output_dir or (Path(reported) if isinstance(reported, str) and reported else comfy_dir / "output")
    object_info = client.get("/object_info"); missing_nodes = sorted(x for x in spec.class_types if x not in object_info)
    if missing_nodes: raise ValidationError(f"ComfyUI missing node types: {missing_nodes}")
    if not skip_models:
        missing = [str(p) for p in model_files(spec, comfy_dir) if not p.is_file()]
        if missing: raise ValidationError(f"Model files missing: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True); probe = output_dir / ".batch_generate_write_test"
    try: probe.write_text("ok", encoding="ascii"); probe.unlink()
    except OSError as exc: raise ValidationError(f"Output directory not writable: {exc}") from exc
    return output_dir


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True); path = log_dir / f"sdxl_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s"); root = logging.getLogger(); root.setLevel(logging.DEBUG if verbose else logging.INFO)
    handlers = [logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers: handler.setFormatter(formatter)
    root.handlers[:] = handlers; return path


def complete_from_history(db: StateDB, row: sqlite3.Row, entry: dict[str, Any], spec: WorkflowSpec, output_dir: Path) -> tuple[bool, str | None]:
    error = extract_execution_error(entry)
    if error: return False, error
    status = entry.get("status", {})
    if not (isinstance(status, dict) and status.get("completed") is True) and not entry.get("outputs"): return False, None
    try: files = extract_outputs(entry, spec.save_node, output_dir)
    except BatchError as exc: return False, str(exc)
    updated = db.update(row, "completed", output_files=files); db.event(updated, "completed", output_files=files)
    logging.info("completed sample=%s size=%s", row["sample_id"], row["size_key"]); return True, None


def row_job(row: sqlite3.Row) -> Job: return Job(row["sample_id"], row["prompt"], row["width"], row["height"], row["size_key"], row["seed"])


def run_jobs(args, jobs, spec, client, db, output_dir, stop) -> dict[str, int]:
    rows = db.select(jobs); stats = {"planned": len(rows), "completed": 0, "failed": 0, "skipped": 0, "retried": 0}
    pending, inflight = deque(), {}; remote_queue = client.get("/queue") if args.resume else {}
    for row in rows:
        if args.resume and row["status"] == "completed" and output_files_exist(row): stats["skipped"] += 1; continue
        if args.resume and row["prompt_id"]:
            entry = client.history(row["prompt_id"])
            if entry:
                complete, error = complete_from_history(db, row, entry, spec, output_dir)
                if complete: stats["completed"] += 1; continue
                if row["status"] in ("submitted", "running") and not error: inflight[row["prompt_id"]] = (row, time.monotonic(), 0); continue
            elif row["status"] in ("submitted", "running") and contains_value(remote_queue, row["prompt_id"]):
                inflight[row["prompt_id"]] = (row, time.monotonic(), 0); continue
        pending.append((db.update(row, "pending"), 0))
    def retry_or_fail(row, failures, exc):
        kind, message = getattr(exc, "kind", classify_error(str(exc))), str(exc)[:12000]; db.event(row, "attempt_failed", kind, message)
        if kind not in {"node_validation", "http_400", "validation"} and failures <= args.max_retries and not stop[0]:
            pending.append((db.update(row, "pending", error_type=kind, error_message=message), failures)); stats["retried"] += 1
            logging.warning("retry sample=%s size=%s type=%s", row["sample_id"], row["size_key"], kind)
            if args.retry_delay: time.sleep(args.retry_delay)
        else:
            updated = db.update(row, "failed", error_type=kind, error_message=message); db.event(updated, "failed", kind, message); stats["failed"] += 1
            logging.error("failed sample=%s size=%s type=%s", row["sample_id"], row["size_key"], kind)
    while (pending or inflight) and not stop[0]:
        while pending and len(inflight) < args.queue_size and not stop[0]:
            try:
                if queue_depth(client.get("/queue")) >= args.queue_size: break
            except BatchError as exc: logging.warning("queue check failed: %s", exc); break
            row, failures = pending.popleft(); workflow = spec.render(
                row_job(row), args.run_name, row["negative_prompt"], args.sampling_preset, args.disable_refiner
            )
            row = db.update(row, "submitting", increment_attempt=True, output_files=[], clear_prompt_id=True)
            try:
                prompt_id = client.submit(workflow, args.client_id); row = db.update(row, "submitted", prompt_id=prompt_id); db.event(row, "submitted")
                inflight[prompt_id] = (row, time.monotonic(), failures); logging.info("submitted sample=%s size=%s id=%s", row["sample_id"], row["size_key"], prompt_id)
            except BatchError as exc: retry_or_fail(row, failures + 1, exc)
        finished = []
        for prompt_id, (row, started, failures) in list(inflight.items()):
            try:
                entry = client.history(prompt_id)
                if entry:
                    complete, error = complete_from_history(db, row, entry, spec, output_dir)
                    if complete: stats["completed"] += 1; finished.append(prompt_id)
                    elif error:
                        exc = ExecutionError(error); exc.kind = classify_error(error); retry_or_fail(row, failures + 1, exc); finished.append(prompt_id)
                elif time.monotonic() - started > args.job_timeout:
                    exc = ExecutionError(f"Timed out after {args.job_timeout}s"); exc.kind = "timeout"; retry_or_fail(row, failures + 1, exc); finished.append(prompt_id)
            except BatchError as exc:
                if getattr(exc, "kind", "") == "transport": logging.warning("poll failed; job remains in flight: %s", exc)
                else: retry_or_fail(row, failures + 1, exc); finished.append(prompt_id)
        for prompt_id in finished: inflight.pop(prompt_id, None)
        if (pending or inflight) and not stop[0]: time.sleep(args.poll_interval)
    if stop[0]: logging.warning("interrupted; %d in-flight jobs remain for --resume", len(inflight))
    return stats


def selected_prompts(prompts, start, limit):
    if start < 0: raise ValidationError("--start must be >=0")
    if limit is not None and limit < 1: raise ValidationError("--limit must be >=1")
    return prompts[start:] if limit is None else prompts[start:start + limit]


def resolve_sizes(args, spec):
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None: raise ValidationError("--width and --height must be paired")
        if args.size_mode == "prompt" or args.sizes: raise ValidationError("Fixed dimensions conflict with prompt mode or --sizes")
        return "fixed", [parse_size(f"{args.width}x{args.height}")]
    if args.sizes:
        if args.size_mode == "prompt": raise ValidationError("--sizes conflicts with --size-mode prompt")
        values = list(dict.fromkeys(parse_size(v) for v in args.sizes.split(",") if v.strip()))
        if not values: raise ValidationError("--sizes is empty")
        return "all", values
    if args.size_mode == "prompt": return "prompt", []
    node = spec.template[spec.dimension_node]["inputs"]; return "workflow", [(node["width"], node["height"])]


def write_reports(db, state_dir, jobs):
    rows, counts, missing = db.select(jobs), {}, []
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] == "completed" and not output_files_exist(row): missing.append({"sample_id": row["sample_id"], "size": row["size_key"]})
    report = {"generated_at": now(), "model": MODEL, "counts": counts, "missing_files": missing}
    atomic_json_dump(state_dir / "summary.json", report); atomic_json_dump(state_dir / "missing_files.json", missing); return report, missing


def build_parser(script_dir: Path):
    packaged = script_dir / "input/t2i_prompt_bank_1500.json"
    repository = script_dir.parent / "data/prompts/v2/remote/t2i_prompt_bank_1500.json"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompts", type=Path, default=packaged if packaged.is_file() else repository); p.add_argument("--workflow", type=Path, default=script_dir / "input/sdxl_api.json")
    p.add_argument("--expected-count", type=int, default=1500); p.add_argument("--comfy-url", default="http://127.0.0.1:18188")
    p.add_argument("--comfy-dir", type=Path, default=Path("/workspace/data_new/ComfyUI")); p.add_argument("--output-dir", type=Path)
    p.add_argument("--state-dir", type=Path, default=script_dir / "state/sdxl"); p.add_argument("--log-dir", type=Path, default=script_dir / "logs")
    p.add_argument("--resume", action="store_true"); p.add_argument("--limit", type=int); p.add_argument("--start", "--start-index", dest="start", type=int, default=0)
    p.add_argument("--smoke-test", type=int, choices=(1,2,3)); p.add_argument("--dry-run", action="store_true"); p.add_argument("--queue-size", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=2); p.add_argument("--retry-delay", type=float, default=5); p.add_argument("--poll-interval", type=float, default=2)
    p.add_argument("--http-timeout", type=float, default=30); p.add_argument("--job-timeout", type=float, default=1800); p.add_argument("--seed-salt", default="t2i-sdxl-v1")
    p.add_argument("--run-name"); p.add_argument("--client-id", default="vast-sdxl-batch-generator"); p.add_argument("--size-mode", choices=("prompt","workflow"), default="prompt")
    p.add_argument("--sizes"); p.add_argument("--width", type=int); p.add_argument("--height", type=int); p.add_argument("--sampling-preset", choices=("quality","workflow"), default="quality")
    p.add_argument("--disable-refiner", action="store_true", help="run the Base sampler to completion and decode with the Base VAE")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT); p.add_argument("--negative-prompt-file", type=Path)
    p.add_argument("--skip-preflight", action="store_true"); p.add_argument("--skip-model-file-check", action="store_true"); p.add_argument("--verbose", action="store_true"); return p


def main(argv=None):
    script_dir = Path(__file__).resolve().parent; parser = build_parser(script_dir); args = parser.parse_args(argv)
    try:
        if args.queue_size < 1 or args.max_retries < 0 or args.expected_count < 0: raise ValidationError("Invalid non-negative/count option")
        if args.run_name and (not SAFE_NAME.fullmatch(args.run_name) or args.run_name in {".",".."}): raise ValidationError("Unsafe --run-name")
        if args.negative_prompt_file:
            if args.negative_prompt != DEFAULT_NEGATIVE_PROMPT: raise ValidationError("Choose --negative-prompt or --negative-prompt-file")
            args.negative_prompt = args.negative_prompt_file.read_text(encoding="utf-8").strip()
        if not args.negative_prompt.strip(): raise ValidationError("Negative prompt is empty")
        prompts = validate_prompts(args.prompts)
        if args.expected_count and len(prompts) != args.expected_count: raise ValidationError(f"Expected {args.expected_count} prompts, found {len(prompts)}")
        prompts = selected_prompts(prompts, args.start, args.smoke_test or args.limit)
        if not prompts: raise ValidationError("Selected range is empty")
        spec = WorkflowSpec.inspect(args.workflow); mode, sizes = resolve_sizes(args, spec); jobs = make_jobs(prompts, mode, sizes, args.seed_salt)
        unique_sizes = sorted({j.size_key for j in jobs}); plan = {"model": MODEL, "prompt_count": len(prompts), "total_jobs": len(jobs),
            "first_sample": prompts[0]["sample_id"], "last_sample": prompts[-1]["sample_id"], "size_mode": mode, "sizes": unique_sizes,
            "size_counts": {s: sum(j.size_key == s for j in jobs) for s in unique_sizes}, "queue_size": args.queue_size,
            "run_name": args.run_name, "workflow": spec.summary(args.sampling_preset, args.negative_prompt, args.disable_refiner)}
        if args.dry_run: print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    except (ValidationError, OSError) as exc: print(f"validation error: {exc}", file=sys.stderr); return 2
    log_path = setup_logging(args.log_dir, args.verbose); logging.info("plan=%s", json.dumps(plan, ensure_ascii=False))
    client, output_dir, stop, db = ComfyClient(args.comfy_url, args.http_timeout), args.output_dir, [False], None
    def on_signal(signum, _frame): logging.warning("received signal %s", signum); stop[0] = True
    signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)
    try:
        output_dir = preflight(client, spec, args.comfy_dir, output_dir, args.skip_model_file_check) if not args.skip_preflight else (output_dir or args.comfy_dir / "output")
        generation_config = json.dumps({"sampling_preset": args.sampling_preset, "disable_refiner": args.disable_refiner}, sort_keys=True)
        db = StateDB(args.state_dir / "jobs.sqlite3"); db.seed_jobs(jobs, args.negative_prompt, generation_config); result = run_jobs(args, jobs, spec, client, db, output_dir, stop)
        report, missing = write_reports(db, args.state_dir, jobs); payload = {"result": result, "report": report, "log": str(log_path)}
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 130 if stop[0] else (1 if missing or result["failed"] else 0)
    except (BatchError, OSError, sqlite3.Error) as exc: logging.exception("fatal error: %s", exc); return 2
    finally:
        if db: db.close()


if __name__ == "__main__": raise SystemExit(main())
