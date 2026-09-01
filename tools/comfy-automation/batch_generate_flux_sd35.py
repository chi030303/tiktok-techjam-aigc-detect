#!/usr/bin/env python3
"""Resumable, bounded-queue ComfyUI batch generator for FLUX.2 and SD 3.5.

Only the Python standard library is required.  Run this script on the same Vast
machine as ComfyUI so output files can be verified directly on disk.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODELS = ("flux2", "sd35")
SAFE_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MODEL_DIRS = {
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "UNETLoader": ("unet_name", "diffusion_models"),
    "CLIPLoader": ("clip_name", "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
}
TERMINAL = {"completed", "failed"}


class BatchError(RuntimeError):
    kind = "internal"


class ValidationError(BatchError):
    kind = "validation"


class ExecutionError(BatchError):
    kind = "execution"


class OutputMissingError(BatchError):
    kind = "missing_output"


class ComfyHTTPError(BatchError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        self.kind = classify_error(body, "http_400" if status == 400 else "http")
        super().__init__(f"ComfyUI HTTP {status}: {body[:1500]}")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def classify_error(message: str, default: str = "execution") -> str:
    text = message.lower()
    if any(x in text for x in ("out of memory", "cuda oom", "cuda error: out")):
        return "oom"
    if any(x in text for x in ("prompt_outputs_failed_validation", "invalid prompt", "required input")):
        return "node_validation"
    return default


def stable_seed(sample_id: str, model: str, salt: str) -> int:
    raw = hashlib.sha256(f"{salt}\0{model}\0{sample_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") & ((1 << 63) - 1)


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc


def validate_prompts(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValidationError("Prompt file must be a JSON array")
    if not data:
        raise ValidationError("Prompt file is empty")
    seen: set[str] = set()
    validated = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(f"Prompt #{index + 1} is not an object")
        sample_id = item.get("prompt_id") or item.get("sample_id") or item.get("id")
        text = item.get("rendered_prompt") or item.get("prompt") or item.get("text")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValidationError(f"Prompt #{index + 1} has no non-empty stable ID")
        sample_id = sample_id.strip()
        if not SAFE_SAMPLE_ID.fullmatch(sample_id) or sample_id in {".", ".."}:
            raise ValidationError(
                f"Prompt ID {sample_id!r} is unsafe for filenames; use 1-128 ASCII letters, digits, dot, underscore or hyphen"
            )
        if sample_id in seen:
            raise ValidationError(f"Duplicate prompt ID: {sample_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError(f"Prompt {sample_id} has empty text")
        seen.add(sample_id)
        normalized = dict(item)
        normalized["sample_id"] = sample_id
        normalized["prompt_text"] = text.strip()
        validated.append(normalized)
    return validated


def unsafe_path_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.startswith(("/Users/", "/home/")) or ":\\" in value:
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from unsafe_path_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from unsafe_path_strings(child)


@dataclass(frozen=True)
class WorkflowSpec:
    model: str
    template: dict[str, Any]
    prompt_node: str
    prompt_field: str
    seed_node: str
    seed_field: str
    save_node: str
    width_target: tuple[str, str] | None
    height_target: tuple[str, str] | None
    class_types: frozenset[str]

    @classmethod
    def inspect(cls, model: str, path: Path) -> "WorkflowSpec":
        workflow = load_json(path)
        if not isinstance(workflow, dict) or not workflow:
            raise ValidationError(f"{model} workflow must be a non-empty API node map")
        for node_id, node in workflow.items():
            if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
                raise ValidationError(f"{model} node {node_id!r} is not ComfyUI API format")
        bad_paths = list(unsafe_path_strings(workflow))
        if bad_paths:
            raise ValidationError(f"{model} workflow contains local absolute paths: {bad_paths}")

        def title(node: dict[str, Any]) -> str:
            return str(node.get("_meta", {}).get("title", "")).lower()

        positive: list[tuple[str, str]] = []
        seed_fields: list[tuple[str, str]] = []
        saves = []
        width_target = height_target = None
        for node_id, node in workflow.items():
            inputs = node["inputs"]
            label = title(node)
            if "negative" not in label and "负向" not in label:
                if isinstance(inputs.get("text"), str) and ("prompt" in label or node["class_type"] == "CLIPTextEncode"):
                    positive.append((node_id, "text"))
                if isinstance(inputs.get("value"), str) and "prompt" in label:
                    positive.append((node_id, "value"))
            for field in ("noise_seed", "seed"):
                if isinstance(inputs.get(field), int):
                    seed_fields.append((node_id, field))
            if node["class_type"] == "SaveImage":
                saves.append(node_id)
            if isinstance(inputs.get("width"), int):
                width_target = (node_id, "width")
            if isinstance(inputs.get("height"), int):
                height_target = (node_id, "height")
            if node["class_type"] == "PrimitiveInt" and isinstance(inputs.get("value"), int):
                if "width" in label:
                    width_target = (node_id, "value")
                if "height" in label:
                    height_target = (node_id, "value")
        positive = list(dict.fromkeys(positive))
        if len(positive) != 1:
            raise ValidationError(f"{model}: expected exactly one positive prompt entry, found {positive}")
        if len(seed_fields) != 1:
            raise ValidationError(f"{model}: expected exactly one seed input, found {seed_fields}")
        if len(saves) != 1:
            raise ValidationError(f"{model}: expected exactly one SaveImage node, found {saves}")
        return cls(
            model=model,
            template=workflow,
            prompt_node=positive[0][0],
            prompt_field=positive[0][1],
            seed_node=seed_fields[0][0],
            seed_field=seed_fields[0][1],
            save_node=saves[0],
            width_target=width_target,
            height_target=height_target,
            class_types=frozenset(node["class_type"] for node in workflow.values()),
        )

    def render(self, sample_id: str, prompt: str, seed: int, width: int | None, height: int | None,
               run_name: str | None = None) -> dict[str, Any]:
        workflow = copy.deepcopy(self.template)
        workflow[self.prompt_node]["inputs"][self.prompt_field] = prompt
        workflow[self.seed_node]["inputs"][self.seed_field] = seed
        prefix = f"{self.model}/{sample_id}_{self.model}"
        if run_name:
            prefix = f"{run_name}/{prefix}"
        workflow[self.save_node]["inputs"]["filename_prefix"] = prefix
        if width is not None:
            if not self.width_target:
                raise ValidationError(f"{self.model} workflow has no mutable width input")
            workflow[self.width_target[0]]["inputs"][self.width_target[1]] = width
        if height is not None:
            if not self.height_target:
                raise ValidationError(f"{self.model} workflow has no mutable height input")
            workflow[self.height_target[0]]["inputs"][self.height_target[1]] = height
        return workflow

    def summary(self) -> dict[str, Any]:
        def input_value(target: tuple[str, str] | None) -> Any:
            return self.template[target[0]]["inputs"][target[1]] if target else None
        model_files = []
        for node in self.template.values():
            mapping = MODEL_DIRS.get(node["class_type"])
            if mapping and mapping[0] in node["inputs"]:
                model_files.append({"class_type": node["class_type"], "file": node["inputs"][mapping[0]]})
        return {
            "prompt": f"{self.prompt_node}.{self.prompt_field}",
            "seed": f"{self.seed_node}.{self.seed_field}",
            "save_node": self.save_node,
            "width": input_value(self.width_target),
            "height": input_value(self.height_target),
            "model_files": model_files,
            "class_types": sorted(self.class_types),
        }


class ComfyClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, payload: Any = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ComfyHTTPError(exc.code, body) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = BatchError(f"Cannot reach ComfyUI at {self.base_url}: {exc}")
            error.kind = "transport"
            raise error from exc
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BatchError(f"Invalid JSON from {path}: {raw[:500]!r}") from exc

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, payload: Any) -> Any:
        return self.request("POST", path, payload)

    def submit(self, workflow: dict[str, Any], client_id: str) -> str:
        response = self.post("/prompt", {"prompt": workflow, "client_id": client_id})
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValidationError(f"/prompt did not return prompt_id: {response}")
        return prompt_id

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        response = self.get("/history/" + urllib.parse.quote(prompt_id, safe=""))
        entry = response.get(prompt_id) if isinstance(response, dict) else None
        return entry if isinstance(entry, dict) else None

    def free(self) -> None:
        self.post("/free", {"unload_models": True, "free_memory": True})


class StateDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              sample_id TEXT NOT NULL, model TEXT NOT NULL, prompt TEXT NOT NULL,
              seed INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
              prompt_id TEXT, attempt INTEGER NOT NULL DEFAULT 0,
              output_files TEXT NOT NULL DEFAULT '[]', error_type TEXT,
              error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              PRIMARY KEY (sample_id, model)
            );
            CREATE TABLE IF NOT EXISTS attempts (
              id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
              model TEXT NOT NULL, attempt INTEGER NOT NULL, prompt_id TEXT,
              status TEXT NOT NULL, error_type TEXT, error_message TEXT,
              output_files TEXT NOT NULL DEFAULT '[]', timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_model_status ON jobs(model, status);
            """
        )

    def seed_jobs(self, prompts: list[dict[str, Any]], models: list[str], salt: str) -> None:
        stamp = now()
        with self.connection:
            for model in models:
                self.connection.executemany(
                    """INSERT OR IGNORE INTO jobs
                       (sample_id, model, prompt, seed, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (p["sample_id"], model, p["prompt_text"], stable_seed(p["sample_id"], model, salt), stamp, stamp)
                        for p in prompts
                    ],
                )
        for model in models:
            existing = {
                row["sample_id"]: row
                for row in self.connection.execute("SELECT sample_id, prompt, seed FROM jobs WHERE model=?", (model,))
            }
            for prompt in prompts:
                row = existing[prompt["sample_id"]]
                expected_seed = stable_seed(prompt["sample_id"], model, salt)
                if row["prompt"] != prompt["prompt_text"] or row["seed"] != expected_seed:
                    raise ValidationError(
                        f"State conflict for ({prompt['sample_id']}, {model}): prompt or seed salt changed; "
                        "use a separate --state-dir or restore the original input"
                    )

    def select(self, model: str, sample_ids: list[str]) -> list[sqlite3.Row]:
        wanted = set(sample_ids)
        rows = self.connection.execute("SELECT * FROM jobs WHERE model=?", (model,)).fetchall()
        by_id = {row["sample_id"]: row for row in rows}
        return [by_id[sample_id] for sample_id in sample_ids if sample_id in wanted]

    def update(self, sample_id: str, model: str, status: str, *, prompt_id: str | None = None,
               output_files: list[str] | None = None, error_type: str | None = None,
               error_message: str | None = None, increment_attempt: bool = False,
               clear_prompt_id: bool = False) -> sqlite3.Row:
        fields = ["status=?", "updated_at=?", "error_type=?", "error_message=?"]
        values: list[Any] = [status, now(), error_type, error_message]
        if clear_prompt_id:
            fields.append("prompt_id=NULL")
        elif prompt_id is not None:
            fields.append("prompt_id=?")
            values.append(prompt_id)
        if output_files is not None:
            fields.append("output_files=?")
            values.append(json.dumps(output_files, ensure_ascii=False))
        if increment_attempt:
            fields.append("attempt=attempt+1")
        values.extend([sample_id, model])
        with self.connection:
            self.connection.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE sample_id=? AND model=?", values)
        return self.connection.execute("SELECT * FROM jobs WHERE sample_id=? AND model=?", (sample_id, model)).fetchone()

    def event(self, row: sqlite3.Row, status: str, *, error_type: str | None = None,
              error_message: str | None = None, output_files: list[str] | None = None) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO attempts
                   (sample_id, model, attempt, prompt_id, status, error_type, error_message, output_files, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["sample_id"], row["model"], row["attempt"], row["prompt_id"], status,
                 error_type, error_message, json.dumps(output_files or [], ensure_ascii=False), now()),
            )

    def close(self) -> None:
        self.connection.close()


def extract_execution_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status", {})
    if isinstance(status, dict) and status.get("status_str") == "error":
        messages = status.get("messages", [])
        return json.dumps(messages, ensure_ascii=False)[:12000] or "ComfyUI execution error"
    for message in status.get("messages", []) if isinstance(status, dict) else []:
        if isinstance(message, list) and message and message[0] == "execution_error":
            return json.dumps(message, ensure_ascii=False)[:12000]
    return None


def extract_outputs(entry: dict[str, Any], save_node: str, output_dir: Path) -> list[str]:
    outputs = entry.get("outputs", {})
    node_output = outputs.get(save_node, {}) if isinstance(outputs, dict) else {}
    images = node_output.get("images", []) if isinstance(node_output, dict) else []
    files: list[str] = []
    root = output_dir.resolve()
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("filename"), str):
            continue
        if image.get("type", "output") != "output":
            continue
        path = (output_dir / str(image.get("subfolder", "")) / image["filename"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"ComfyUI returned output outside output directory: {path}") from exc
        files.append(str(path))
    if not files:
        raise OutputMissingError(f"History contains no images for SaveImage node {save_node}")
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise OutputMissingError(f"History reports files that do not exist: {missing}")
    return files


def output_files_exist(row: sqlite3.Row) -> bool:
    try:
        files = json.loads(row["output_files"])
    except json.JSONDecodeError:
        return False
    return bool(files) and all(Path(path).is_file() for path in files)


def contains_value(value: Any, needle: str) -> bool:
    if value == needle:
        return True
    if isinstance(value, dict):
        return any(contains_value(child, needle) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_value(child, needle) for child in value)
    return False


def model_files(spec: WorkflowSpec, comfy_dir: Path) -> list[Path]:
    found = []
    for node in spec.template.values():
        mapping = MODEL_DIRS.get(node["class_type"])
        if mapping and isinstance(node["inputs"].get(mapping[0]), str):
            found.append(comfy_dir / "models" / mapping[1] / node["inputs"][mapping[0]])
    return found


def preflight(client: ComfyClient, specs: dict[str, WorkflowSpec], comfy_dir: Path, output_dir: Path | None,
              skip_model_files: bool) -> Path:
    stats = client.get("/system_stats")
    logging.info("ComfyUI system_stats received; devices=%s", stats.get("devices", "unknown"))
    system = stats.get("system", {}) if isinstance(stats, dict) else {}
    reported_output = system.get("output_directory") if isinstance(system, dict) else None
    if not reported_output and isinstance(stats, dict):
        reported_output = stats.get("output_directory")
    if output_dir is None:
        output_dir = Path(reported_output) if isinstance(reported_output, str) and reported_output else comfy_dir / "output"
        logging.info("using ComfyUI-reported output directory: %s", output_dir)
    elif reported_output and output_dir.resolve() != Path(reported_output).resolve():
        logging.warning("explicit --output-dir %s differs from ComfyUI output_directory %s", output_dir, reported_output)
    object_info = client.get("/object_info")
    missing_nodes = sorted({name for spec in specs.values() for name in spec.class_types if name not in object_info})
    if missing_nodes:
        raise ValidationError(f"ComfyUI is missing workflow node types: {missing_nodes}")
    if not skip_model_files:
        missing_models = [str(path) for spec in specs.values() for path in model_files(spec, comfy_dir) if not path.is_file()]
        if missing_models:
            raise ValidationError(f"Model files missing on server: {missing_models}")
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".batch_generate_write_test"
    try:
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
    except OSError as exc:
        raise ValidationError(f"Output directory is not writable: {output_dir}: {exc}") from exc
    return output_dir


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"batch_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.handlers[:] = [file_handler, stream_handler]
    return path


def selected_prompts(prompts: list[dict[str, Any]], start: int, limit: int | None) -> list[dict[str, Any]]:
    if start < 0:
        raise ValidationError("--start must be >= 0")
    if limit is not None and limit < 1:
        raise ValidationError("--limit must be >= 1")
    return prompts[start:] if limit is None else prompts[start:start + limit]


def parse_dimensions(args: argparse.Namespace) -> tuple[int | None, int | None]:
    if (args.width is None) != (args.height is None):
        raise ValidationError("--width and --height must be supplied together")
    for name, value in (("width", args.width), ("height", args.height)):
        if value is not None and (value < 64 or value % 8):
            raise ValidationError(f"--{name} must be >= 64 and divisible by 8")
    return args.width, args.height


def complete_from_history(db: StateDB, row: sqlite3.Row, entry: dict[str, Any], spec: WorkflowSpec,
                          output_dir: Path) -> tuple[bool, str | None]:
    error = extract_execution_error(entry)
    if error:
        return False, error
    status = entry.get("status", {})
    completed = isinstance(status, dict) and status.get("completed") is True
    if not completed and not entry.get("outputs"):
        return False, None
    try:
        files = extract_outputs(entry, spec.save_node, output_dir)
    except BatchError as exc:
        return False, str(exc)
    updated = db.update(row["sample_id"], row["model"], "completed", output_files=files)
    db.event(updated, "completed", output_files=files)
    logging.info("completed model=%s sample=%s files=%s", row["model"], row["sample_id"], files)
    return True, None


def run_model(args: argparse.Namespace, model: str, prompts: list[dict[str, Any]], spec: WorkflowSpec,
              client: ComfyClient, db: StateDB, output_dir: Path, stop: list[bool]) -> dict[str, Any]:
    rows = db.select(model, [p["sample_id"] for p in prompts])
    stats: dict[str, Any] = {"model": model, "planned": len(rows), "completed": 0, "failed": 0, "skipped": 0}
    pending: deque[tuple[sqlite3.Row, int]] = deque()
    inflight: dict[str, tuple[sqlite3.Row, float, int]] = {}

    active = [row for row in rows if args.resume and row["status"] in ("submitted", "running") and row["prompt_id"]]
    remote_queue = client.get("/queue") if active else {}
    for row in rows:
        if args.resume and row["status"] == "completed" and output_files_exist(row):
            stats["skipped"] += 1
            continue
        if row["status"] == "completed" and not output_files_exist(row):
            logging.warning("completed record has missing files; regenerating model=%s sample=%s", model, row["sample_id"])
        if args.resume and row["prompt_id"]:
            entry = client.history(row["prompt_id"])
            if entry:
                complete, error = complete_from_history(db, row, entry, spec, output_dir)
                if complete:
                    stats["completed"] += 1
                    logging.info("recovered prior ComfyUI result model=%s sample=%s", model, row["sample_id"])
                    continue
                if row["status"] in ("submitted", "running") and not error:
                    inflight[row["prompt_id"]] = (row, time.monotonic(), 0)
                else:
                    pending.append((db.update(row["sample_id"], model, "pending"), 0))
            elif row["status"] in ("submitted", "running") and contains_value(remote_queue, row["prompt_id"]):
                inflight[row["prompt_id"]] = (row, time.monotonic(), 0)
            else:
                logging.warning("prior job is absent from history/queue; resubmitting model=%s sample=%s", model, row["sample_id"])
                pending.append((db.update(row["sample_id"], model, "pending"), 0))
        else:
            pending.append((row, 0))

    def retry_or_fail(row: sqlite3.Row, run_failures: int, exc: BaseException) -> None:
        kind = getattr(exc, "kind", classify_error(str(exc)))
        message = str(exc)[:12000]
        db.event(row, "attempt_failed", error_type=kind, error_message=message)
        if run_failures <= args.max_retries and not stop[0]:
            updated = db.update(row["sample_id"], model, "pending", error_type=kind, error_message=message)
            logging.warning("retrying model=%s sample=%s failure=%s/%s type=%s", model, row["sample_id"], run_failures, args.max_retries, kind)
            if args.retry_delay:
                time.sleep(args.retry_delay)
            pending.append((updated, run_failures))
        else:
            updated = db.update(row["sample_id"], model, "failed", error_type=kind, error_message=message)
            db.event(updated, "failed", error_type=kind, error_message=message)
            stats["failed"] += 1
            logging.error("failed model=%s sample=%s type=%s error=%s", model, row["sample_id"], kind, message)

    while (pending or inflight) and not stop[0]:
        while pending and len(inflight) < args.queue_size and not stop[0]:
            row, run_failures = pending.popleft()
            workflow = spec.render(
                row["sample_id"], row["prompt"], row["seed"], args.width, args.height,
                getattr(args, "run_name", None),
            )
            row = db.update(
                row["sample_id"], model, "submitting", increment_attempt=True,
                error_type=None, error_message=None, output_files=[], clear_prompt_id=True,
            )
            try:
                prompt_id = client.submit(workflow, args.client_id)
                row = db.update(row["sample_id"], model, "submitted", prompt_id=prompt_id)
                db.event(row, "submitted")
                inflight[prompt_id] = (row, time.monotonic(), run_failures)
                logging.info("submitted model=%s sample=%s prompt_id=%s attempt=%s", model, row["sample_id"], prompt_id, row["attempt"])
            except BatchError as exc:
                retry_or_fail(row, run_failures + 1, exc)

        finished: list[str] = []
        for prompt_id, (row, started, run_failures) in list(inflight.items()):
            try:
                entry = client.history(prompt_id)
                if entry:
                    complete, error = complete_from_history(db, row, entry, spec, output_dir)
                    if complete:
                        stats["completed"] += 1
                        finished.append(prompt_id)
                    elif error:
                        exc = ExecutionError(error)
                        exc.kind = classify_error(error)
                        retry_or_fail(row, run_failures + 1, exc)
                        finished.append(prompt_id)
                elif time.monotonic() - started > args.job_timeout:
                    exc = ExecutionError(f"Timed out after {args.job_timeout}s waiting for {prompt_id}")
                    exc.kind = "timeout"
                    retry_or_fail(row, run_failures + 1, exc)
                    finished.append(prompt_id)
            except BatchError as exc:
                if getattr(exc, "kind", "") == "transport":
                    logging.warning("poll failed; leaving job in flight: %s", exc)
                else:
                    retry_or_fail(row, run_failures + 1, exc)
                    finished.append(prompt_id)
        for prompt_id in finished:
            inflight.pop(prompt_id, None)
        if (pending or inflight) and not stop[0]:
            time.sleep(args.poll_interval)

    if stop[0]:
        logging.warning("interrupted; %d remote jobs remain recorded for --resume", len(inflight))
    return stats


def write_reports(db: StateDB, state_dir: Path, models: list[str], selected_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    placeholders = ",".join("?" for _ in selected_ids)
    rows = db.connection.execute(
        f"SELECT * FROM jobs WHERE model IN ({','.join('?' for _ in models)}) AND sample_id IN ({placeholders})",
        [*models, *selected_ids],
    ).fetchall()
    counts: dict[str, dict[str, int]] = {}
    missing = []
    for row in rows:
        counts.setdefault(row["model"], {}).setdefault(row["status"], 0)
        counts[row["model"]][row["status"]] += 1
        if row["status"] == "completed" and not output_files_exist(row):
            missing.append({"sample_id": row["sample_id"], "model": row["model"], "output_files": row["output_files"]})
    report = {"generated_at": now(), "counts": counts, "missing_files": missing}
    atomic_json_dump(state_dir / "summary.json", report)
    atomic_json_dump(state_dir / "missing_files.json", missing)
    return report, missing


def build_parser(script_dir: Path) -> argparse.ArgumentParser:
    packaged_prompts = script_dir / "input/prompts.json"
    local_prompts = script_dir.parent / "data/prompts/t2i_prompt_bank_1500.json"
    default_prompts = packaged_prompts if packaged_prompts.is_file() else local_prompts
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--prompts", type=Path, default=default_prompts)
    parser.add_argument("--expected-count", type=int, default=1500, help="fail if the complete prompt file count differs; 0 disables")
    parser.add_argument("--flux-workflow", type=Path, default=script_dir / "workflows/flux2_api.json")
    parser.add_argument("--sd35-workflow", type=Path, default=script_dir / "workflows/sd35_api.json")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:18188")
    parser.add_argument("--comfy-dir", type=Path, default=Path("/workspace/data_new/ComfyUI"))
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to COMFY_DIR/output")
    parser.add_argument("--state-dir", type=Path, default=script_dir / "state")
    parser.add_argument("--log-dir", type=Path, default=script_dir / "logs")
    parser.add_argument("--resume", action="store_true", help="skip completed jobs with existing files and recover submitted jobs")
    parser.add_argument("--limit", type=int, help="process at most N prompts per selected model")
    parser.add_argument("--start", type=int, default=0, help="zero-based prompt offset, useful for explicit batches")
    parser.add_argument("--smoke-test", type=int, choices=(1, 2, 3), help="test the first 1-3 selected prompts per model")
    parser.add_argument("--dry-run", action="store_true", help="validate and show plan without contacting ComfyUI or writing state")
    parser.add_argument("--queue-size", type=int, default=1, help="maximum submitted/in-flight ComfyUI prompts")
    parser.add_argument("--max-retries", type=int, default=2, help="retries after the initial attempt")
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--seed-salt", default="t2i-batch-v1")
    parser.add_argument("--run-name", help="safe output subdirectory, e.g. remote_v2")
    parser.add_argument("--client-id", default="vast-batch-generator")
    parser.add_argument("--width", type=int, help="optional override; paired with --height")
    parser.add_argument("--height", type=int, help="optional override; paired with --width")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-model-file-check", action="store_true")
    parser.add_argument("--no-free-between-models", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = build_parser(script_dir)
    args = parser.parse_args(argv)
    if args.queue_size < 1:
        parser.error("--queue-size must be >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    try:
        args.width, args.height = parse_dimensions(args)
        if args.run_name and (not SAFE_SAMPLE_ID.fullmatch(args.run_name) or args.run_name in {".", ".."}):
            raise ValidationError("--run-name must use 1-128 ASCII letters, digits, dot, underscore or hyphen")
        prompts = validate_prompts(args.prompts)
        if args.expected_count < 0:
            raise ValidationError("--expected-count must be >= 0")
        if args.expected_count and len(prompts) != args.expected_count:
            raise ValidationError(f"Expected {args.expected_count} prompts but found {len(prompts)}")
        limit = args.smoke_test if args.smoke_test else args.limit
        prompts = selected_prompts(prompts, args.start, limit)
        if not prompts:
            raise ValidationError("Selected prompt range is empty")
        models = list(MODELS if args.model == "all" else (args.model,))
        workflow_paths = {"flux2": args.flux_workflow, "sd35": args.sd35_workflow}
        specs = {model: WorkflowSpec.inspect(model, workflow_paths[model]) for model in models}
        plan = {
            "prompt_count": len(prompts), "models": models, "total_jobs": len(prompts) * len(models),
            "start": args.start, "limit": limit, "queue_size": args.queue_size,
            "run_name": args.run_name,
            "dimensions_override": [args.width, args.height],
            "workflows": {model: spec.summary() for model, spec in specs.items()},
            "first_sample": prompts[0]["sample_id"], "last_sample": prompts[-1]["sample_id"],
        }
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
    except ValidationError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2

    log_path = setup_logging(args.log_dir, args.verbose)
    logging.info("plan=%s", json.dumps(plan, ensure_ascii=False))
    output_dir = args.output_dir
    client = ComfyClient(args.comfy_url, args.http_timeout)
    stop = [False]

    def on_signal(signum: int, _frame: Any) -> None:
        logging.warning("received signal %s; stopping after current poll", signum)
        stop[0] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    db: StateDB | None = None
    try:
        if not args.skip_preflight:
            output_dir = preflight(client, specs, args.comfy_dir, output_dir, args.skip_model_file_check)
        elif output_dir is None:
            output_dir = args.comfy_dir / "output"
        db = StateDB(args.state_dir / "jobs.sqlite3")
        db.seed_jobs(prompts, models, args.seed_salt)
        results = []
        for index, model in enumerate(models):
            if stop[0]:
                break
            results.append(run_model(args, model, prompts, specs[model], client, db, output_dir, stop))
            if index < len(models) - 1 and not args.no_free_between_models and not stop[0]:
                logging.info("freeing ComfyUI models before switching model groups")
                client.free()
                time.sleep(2)
        report, missing = write_reports(db, args.state_dir, models, [p["sample_id"] for p in prompts])
        logging.info("run_results=%s final_report=%s", json.dumps(results), json.dumps(report))
        print(json.dumps({"results": results, "report": report, "log": str(log_path)}, ensure_ascii=False, indent=2))
        return 130 if stop[0] else (1 if missing or any(r["failed"] for r in results) else 0)
    except (BatchError, OSError, sqlite3.Error) as exc:
        logging.exception("fatal error: %s", exc)
        return 2
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
