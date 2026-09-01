#!/usr/bin/env python3
"""Resumable Vertex AI Nano Banana image generator for a JSON prompt bank.

Designed for execution on a remote Linux machine. Authentication uses Google
Application Default Credentials; no API key or service-account secret is stored
in the script, state database, or logs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import logging
import mimetypes
import os
import random
import re
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-3.1-flash-lite-image"
DEFAULT_PRICE_PER_IMAGE = 0.0336
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SUPPORTED_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
RATIO_RE = re.compile(r"(?<!\d)(1:1|2:3|3:2|3:4|4:3|4:5|5:4|9:16|16:9|21:9)(?!\d)")
THREAD_LOCAL = threading.local()


class InputError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    sample_id: str
    text: str
    prompt_sha256: str
    aspect_ratio: str


@dataclass
class GenerationResult:
    sample_id: str
    ok: bool
    output_file: str | None
    response_id: str | None
    attempts: list[dict[str, Any]]
    error_type: str | None = None
    error_message: str | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_prompts(path: Path, expected_count: int) -> list[Prompt]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read prompt JSON {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise InputError("Prompt JSON must be a non-empty array")
    if expected_count and len(data) != expected_count:
        raise InputError(f"Expected {expected_count} prompts but found {len(data)}")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise InputError(f"Prompt #{index} is not an object")
        sample_id = item.get("prompt_id") or item.get("sample_id") or item.get("id")
        text = item.get("rendered_prompt") or item.get("prompt") or item.get("text")
        if not isinstance(sample_id, str) or not SAFE_ID.fullmatch(sample_id) or sample_id in {".", ".."}:
            raise InputError(f"Prompt #{index} has an unsafe or missing ID: {sample_id!r}")
        if sample_id in seen:
            raise InputError(f"Duplicate prompt ID: {sample_id}")
        if not isinstance(text, str) or not text.strip():
            raise InputError(f"Prompt {sample_id} has empty text")
        ratio_source = str(item.get("aspect_ratio", ""))
        match = RATIO_RE.search(ratio_source)
        aspect_ratio = match.group(1) if match else "1:1"
        if aspect_ratio not in SUPPORTED_RATIOS:
            raise InputError(f"Unsupported aspect ratio for {sample_id}: {aspect_ratio}")
        text = text.strip()
        declared_hash = item.get("prompt_sha256")
        actual_hash = sha256_text(text)
        if declared_hash is not None and declared_hash != actual_hash:
            raise InputError(f"prompt_sha256 mismatch for {sample_id}")
        seen.add(sample_id)
        prompts.append(Prompt(sample_id, text, actual_hash, aspect_ratio))
    return prompts


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
              sample_id TEXT PRIMARY KEY,
              prompt TEXT NOT NULL,
              prompt_sha256 TEXT NOT NULL,
              aspect_ratio TEXT NOT NULL,
              model TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              attempt INTEGER NOT NULL DEFAULT 0,
              output_file TEXT,
              response_id TEXT,
              error_type TEXT,
              error_message TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              sample_id TEXT NOT NULL,
              attempt INTEGER NOT NULL,
              status TEXT NOT NULL,
              error_type TEXT,
              error_message TEXT,
              response_id TEXT,
              timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_nano_jobs_status ON jobs(status);
            """
        )

    def seed(self, prompts: list[Prompt], model: str) -> None:
        stamp = utc_now()
        with self.connection:
            self.connection.executemany(
                """INSERT OR IGNORE INTO jobs
                   (sample_id, prompt, prompt_sha256, aspect_ratio, model, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(p.sample_id, p.text, p.prompt_sha256, p.aspect_ratio, model, stamp, stamp) for p in prompts],
            )
        existing = {
            row["sample_id"]: row
            for row in self.connection.execute(
                "SELECT sample_id,prompt_sha256,aspect_ratio,model FROM jobs"
            )
        }
        for prompt in prompts:
            row = existing[prompt.sample_id]
            if (
                row["prompt_sha256"] != prompt.prompt_sha256
                or row["aspect_ratio"] != prompt.aspect_ratio
                or row["model"] != model
            ):
                raise InputError(
                    f"State conflict for {prompt.sample_id}: prompt, ratio, or model changed; use another --state-dir"
                )

    def rows(self, sample_ids: list[str]) -> list[sqlite3.Row]:
        by_id = {row["sample_id"]: row for row in self.connection.execute("SELECT * FROM jobs")}
        return [by_id[sample_id] for sample_id in sample_ids]

    def mark_running(self, sample_id: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE jobs SET status='running', error_type=NULL, error_message=NULL,
                   updated_at=? WHERE sample_id=?""",
                (utc_now(), sample_id),
            )

    def finish(self, result: GenerationResult) -> None:
        stamp = utc_now()
        with self.connection:
            for event in result.attempts:
                self.connection.execute(
                    """INSERT INTO attempts
                       (sample_id,attempt,status,error_type,error_message,response_id,timestamp)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        result.sample_id,
                        event["attempt"],
                        event["status"],
                        event.get("error_type"),
                        event.get("error_message"),
                        event.get("response_id"),
                        event["timestamp"],
                    ),
                )
            self.connection.execute(
                """UPDATE jobs SET status=?, attempt=attempt+?, output_file=?, response_id=?,
                   error_type=?, error_message=?, updated_at=? WHERE sample_id=?""",
                (
                    "completed" if result.ok else "failed",
                    len(result.attempts),
                    result.output_file,
                    result.response_id,
                    result.error_type,
                    result.error_message,
                    stamp,
                    result.sample_id,
                ),
            )

    def summary(self) -> dict[str, int]:
        return {
            row["status"]: row["count"]
            for row in self.connection.execute("SELECT status,COUNT(*) AS count FROM jobs GROUP BY status")
        }

    def close(self) -> None:
        self.connection.close()


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"nano_banana_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers[:] = [file_handler, stream_handler]
    return path


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def classify_exception(exc: BaseException) -> tuple[str, bool]:
    message = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429 or "429" in message or "resource_exhausted" in message:
        return "rate_limit", True
    if code in {500, 502, 503, 504} or any(x in message for x in ("unavailable", "deadline", "timeout")):
        return "server_or_timeout", True
    if code in {401, 403} or any(x in message for x in ("permission_denied", "unauthenticated")):
        return "authentication", False
    if code == 400 or "invalid_argument" in message:
        return "validation", False
    if any(x in message for x in ("safety", "blocked", "prohibited")):
        return "safety_block", False
    return "api_error", True


def client_for_thread(backend: str, project: str | None, location: str):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    key = (backend, project, location, sha256_text(api_key) if api_key else None)
    if getattr(THREAD_LOCAL, "key", None) != key:
        try:
            from google import genai
        except ImportError as exc:
            raise InputError("Missing dependency: run `python3 -m pip install -U google-genai`") from exc
        if backend == "developer":
            if not api_key:
                raise InputError("Set GEMINI_API_KEY in the environment; do not pass the key on the command line")
            THREAD_LOCAL.client = genai.Client(api_key=api_key)
        else:
            if not project:
                raise InputError("Vertex backend requires --project or GOOGLE_CLOUD_PROJECT")
            THREAD_LOCAL.client = genai.Client(vertexai=True, project=project, location=location)
        THREAD_LOCAL.key = key
    return THREAD_LOCAL.client


def response_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None)
    if parts:
        return list(parts)
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        return list(getattr(content, "parts", None) or [])
    return []


def output_extension(mime_type: str | None) -> str:
    known = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
    return known.get(mime_type or "", mimetypes.guess_extension(mime_type or "") or ".png")


def generate_one(
    prompt: Prompt,
    *,
    backend: str,
    project: str | None,
    location: str,
    model: str,
    output_dir: Path,
    max_retries: int,
    retry_base: float,
    stop: list[bool],
) -> GenerationResult:
    events: list[dict[str, Any]] = []
    for attempt in range(1, max_retries + 2):
        if stop[0]:
            return GenerationResult(prompt.sample_id, False, None, None, events, "interrupted", "Stopped by signal")
        stamp = utc_now()
        try:
            from google.genai import types

            client = client_for_thread(backend, project, location)
            response = client.models.generate_content(
                model=model,
                contents=prompt.text,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    candidate_count=1,
                    image_config=types.ImageConfig(aspect_ratio=prompt.aspect_ratio),
                ),
            )
            image_part = next((part for part in response_parts(response) if getattr(part, "inline_data", None)), None)
            if image_part is None:
                feedback = getattr(response, "prompt_feedback", None)
                error = RuntimeError(f"Model returned no image; prompt_feedback={feedback!r}")
                error.error_kind = "no_image"
                raise error
            inline_data = image_part.inline_data
            image_bytes = inline_data.data
            if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
                raise RuntimeError("Model returned an empty image payload")
            extension = output_extension(getattr(inline_data, "mime_type", None))
            output_file = output_dir / f"{prompt.sample_id}_nano_banana{extension}"
            atomic_write(output_file, bytes(image_bytes))
            response_id = getattr(response, "response_id", None)
            events.append({"attempt": attempt, "status": "completed", "response_id": response_id, "timestamp": stamp})
            return GenerationResult(prompt.sample_id, True, str(output_file.resolve()), response_id, events)
        except BaseException as exc:  # SDK exceptions vary by release; classify by code and message.
            kind = getattr(exc, "error_kind", None)
            retryable = False
            if not kind:
                kind, retryable = classify_exception(exc)
            message = str(exc)[:12000]
            events.append(
                {"attempt": attempt, "status": "attempt_failed", "error_type": kind,
                 "error_message": message, "timestamp": stamp}
            )
            if not retryable or attempt > max_retries or stop[0]:
                return GenerationResult(prompt.sample_id, False, None, None, events, kind, message)
            delay = retry_base * (2 ** (attempt - 1)) + random.uniform(0, min(1.0, retry_base))
            time.sleep(delay)
    raise AssertionError("unreachable")


def is_valid_existing(row: sqlite3.Row) -> bool:
    output = row["output_file"]
    return row["status"] == "completed" and isinstance(output, str) and Path(output).is_file() and Path(output).stat().st_size > 0


def recoverable_output(output_dir: Path, sample_id: str) -> Path | None:
    candidates = sorted(output_dir.glob(f"{sample_id}_nano_banana.*"))
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)


def write_summary(path: Path, db: StateDB, backend: str, model: str, project: str | None,
                  output_dir: Path, log_path: Path) -> dict[str, Any]:
    missing = [
        {"sample_id": row["sample_id"], "output_file": row["output_file"]}
        for row in db.connection.execute("SELECT sample_id,output_file,status FROM jobs WHERE status='completed'")
        if not row["output_file"] or not Path(row["output_file"]).is_file()
    ]
    summary = {
        "generated_at": utc_now(),
        "backend": backend,
        "project": project,
        "model": model,
        "output_dir": str(output_dir.resolve()),
        "counts": db.summary(),
        "missing_files": missing,
        "log": str(log_path.resolve()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return summary


def build_parser(script_dir: Path) -> argparse.ArgumentParser:
    default_prompts = script_dir / "input/t2i_prompt_bank_1500.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=default_prompts)
    parser.add_argument(
        "--backend", choices=("developer", "vertex"),
        default="developer" if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else "vertex",
        help="developer uses GEMINI_API_KEY; vertex uses Application Default Credentials",
    )
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "output/nano_banana/remote_v2")
    parser.add_argument("--state-dir", type=Path, default=script_dir / "state/nano_banana_remote_v2")
    parser.add_argument("--log-dir", type=Path, default=script_dir / "logs")
    parser.add_argument("--expected-count", type=int, default=1500, help="0 disables the full-file count check")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-test", type=int, choices=(1, 2, 3))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--price-per-image", type=float, default=DEFAULT_PRICE_PER_IMAGE)
    parser.add_argument("--max-cost-usd", type=float, help="refuse when estimated selected-job cost exceeds this value")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    args = build_parser(script_dir).parse_args(argv)
    try:
        if args.backend == "vertex" and not args.project:
            raise InputError("Vertex backend requires --project or GOOGLE_CLOUD_PROJECT")
        if args.backend == "developer" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) and not args.dry_run:
            raise InputError("Developer backend requires GEMINI_API_KEY in the environment")
        if args.expected_count < 0 or args.start < 0 or args.workers < 1 or args.max_retries < 0:
            raise InputError("expected-count/start/workers/max-retries values are invalid")
        if args.limit is not None and args.limit < 1:
            raise InputError("--limit must be >= 1")
        prompts = load_prompts(args.prompts, args.expected_count)
        limit = args.smoke_test or args.limit
        prompts = prompts[args.start:] if limit is None else prompts[args.start:args.start + limit]
        if not prompts:
            raise InputError("Selected prompt range is empty")
        estimate = len(prompts) * args.price_per_image
        plan = {
            "backend": args.backend,
            "project": args.project,
            "location": args.location,
            "model": args.model,
            "prompt_count": len(prompts),
            "first_sample": prompts[0].sample_id,
            "last_sample": prompts[-1].sample_id,
            "aspect_ratios": {ratio: sum(p.aspect_ratio == ratio for p in prompts) for ratio in sorted(SUPPORTED_RATIOS) if any(p.aspect_ratio == ratio for p in prompts)},
            "workers": args.workers,
            "price_per_image_usd": args.price_per_image,
            "estimated_max_cost_usd": round(estimate, 2),
            "output_dir": str(args.output_dir),
            "state_dir": str(args.state_dir),
        }
        if args.max_cost_usd is not None and estimate > args.max_cost_usd:
            raise InputError(f"Estimated cost ${estimate:.2f} exceeds --max-cost-usd ${args.max_cost_usd:.2f}")
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
    except InputError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2

    log_path = setup_logging(args.log_dir, args.verbose)
    logging.info("plan=%s", json.dumps(plan, ensure_ascii=False))
    stop = [False]

    def stop_handler(signum: int, _frame: Any) -> None:
        logging.warning("received signal %s; stopping new work", signum)
        stop[0] = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    db: StateDB | None = None
    try:
        # Fail early on missing SDK or credentials before creating 1500 requests.
        client_for_thread(args.backend, args.project, args.location)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        db = StateDB(args.state_dir / "jobs.sqlite3")
        db.seed(prompts, args.model)
        rows = db.rows([p.sample_id for p in prompts])
        pending: list[Prompt] = []
        skipped = 0
        for prompt, row in zip(prompts, rows):
            if args.resume and is_valid_existing(row):
                skipped += 1
            elif (recovered_file := recoverable_output(args.output_dir, prompt.sample_id)) is not None and row["prompt_sha256"] == prompt.prompt_sha256:
                # A crash can occur after atomic image save but before DB commit. Do not charge twice.
                recovered = GenerationResult(
                    prompt.sample_id, True,
                    str(recovered_file.resolve()),
                    row["response_id"],
                    [{"attempt": 0, "status": "recovered", "timestamp": utc_now()}],
                )
                db.finish(recovered)
                skipped += 1
            else:
                pending.append(prompt)
        remaining_estimate = len(pending) * args.price_per_image
        logging.info("pending=%d skipped=%d remaining_estimated_cost_usd=%.2f", len(pending), skipped, remaining_estimate)

        completed = failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="nano-banana") as executor:
            future_to_prompt: dict[concurrent.futures.Future[GenerationResult], Prompt] = {}
            iterator = iter(pending)

            def fill() -> None:
                while not stop[0] and len(future_to_prompt) < args.workers:
                    try:
                        prompt = next(iterator)
                    except StopIteration:
                        return
                    db.mark_running(prompt.sample_id)
                    future = executor.submit(
                        generate_one,
                        prompt,
                        backend=args.backend,
                        project=args.project,
                        location=args.location,
                        model=args.model,
                        output_dir=args.output_dir,
                        max_retries=args.max_retries,
                        retry_base=args.retry_base,
                        stop=stop,
                    )
                    future_to_prompt[future] = prompt

            fill()
            while future_to_prompt:
                done, _ = concurrent.futures.wait(
                    future_to_prompt, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    prompt = future_to_prompt.pop(future)
                    try:
                        result = future.result()
                    except BaseException as exc:
                        kind, _ = classify_exception(exc)
                        result = GenerationResult(
                            prompt.sample_id, False, None, None,
                            [{"attempt": 1, "status": "attempt_failed", "error_type": kind,
                              "error_message": str(exc)[:12000], "timestamp": utc_now()}],
                            kind, str(exc)[:12000],
                        )
                    db.finish(result)
                    if result.ok:
                        completed += 1
                        logging.info("completed sample=%s file=%s", result.sample_id, result.output_file)
                    else:
                        failed += 1
                        logging.error("failed sample=%s type=%s error=%s", result.sample_id, result.error_type, result.error_message)
                fill()

        summary = write_summary(
            args.state_dir / "summary.json", db, args.backend, args.model,
            args.project, args.output_dir, log_path,
        )
        result = {"completed_this_run": completed, "failed_this_run": failed, "skipped": skipped, "summary": summary}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 130 if stop[0] else (1 if failed or summary["missing_files"] else 0)
    except (InputError, OSError, sqlite3.Error) as exc:
        logging.exception("fatal error: %s", exc)
        return 2
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
