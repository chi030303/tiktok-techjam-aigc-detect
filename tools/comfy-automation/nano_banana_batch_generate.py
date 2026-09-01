#!/usr/bin/env python3
"""Resumable Vertex AI Batch image generation through Cloud Storage.

This script deliberately does not use GEMINI_API_KEY or
generativelanguage.googleapis.com. It authenticates with Google Application
Default Credentials, writes Vertex Batch JSONL to GCS, calls
aiplatform.googleapis.com, downloads completed JSONL results, and records every
local image in SQLite.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import logging
import os
import signal
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from nano_banana_generate import (
    DEFAULT_MODEL,
    InputError,
    Prompt,
    atomic_write,
    load_prompts,
    output_extension,
    recoverable_output,
    setup_logging,
    sha256_text,
    utc_now,
)


BATCH_PRICE_PER_IMAGE = 0.0168
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise InputError(f"Expected a gs:// URI, got {uri!r}")
    remainder = uri[5:]
    bucket, separator, prefix = remainder.partition("/")
    if not bucket or any(char.isspace() for char in bucket):
        raise InputError(f"Invalid Cloud Storage bucket URI: {uri!r}")
    return bucket, prefix.strip("/") if separator else ""


def join_gs(root: str, *parts: str) -> str:
    bucket, prefix = parse_gs_uri(root)
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    path = "/".join(part for part in (prefix, suffix) if part)
    return f"gs://{bucket}/{path}" if path else f"gs://{bucket}"


class BatchDB:
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
              batch_id INTEGER,
              output_file TEXT,
              error_type TEXT,
              error_message TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              display_name TEXT NOT NULL UNIQUE,
              job_name TEXT,
              status TEXT NOT NULL,
              api_state TEXT,
              sample_ids TEXT NOT NULL,
              request_count INTEGER NOT NULL DEFAULT 0,
              successful_count INTEGER NOT NULL DEFAULT 0,
              failed_count INTEGER NOT NULL DEFAULT 0,
              pending_count INTEGER NOT NULL DEFAULT 0,
              gcs_input_uri TEXT,
              gcs_output_prefix TEXT,
              gcs_result_uri TEXT,
              error_message TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vertex_batch_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_vertex_batches_status ON batches(status);
            """
        )

    def seed(self, prompts: list[Prompt], model: str, output_dir: Path) -> None:
        stamp = utc_now()
        with self.connection:
            self.connection.executemany(
                """INSERT OR IGNORE INTO jobs
                   (sample_id,prompt,prompt_sha256,aspect_ratio,model,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                [(p.sample_id, p.text, p.prompt_sha256, p.aspect_ratio, model, stamp, stamp) for p in prompts],
            )
        by_id = {row["sample_id"]: row for row in self.connection.execute("SELECT * FROM jobs")}
        with self.connection:
            for prompt in prompts:
                row = by_id[prompt.sample_id]
                if (
                    row["prompt_sha256"] != prompt.prompt_sha256
                    or row["aspect_ratio"] != prompt.aspect_ratio
                    or row["model"] != model
                ):
                    raise InputError(
                        f"State conflict for {prompt.sample_id}; use a separate --state-dir for changed input/model"
                    )
                if row["status"] == "completed":
                    output = Path(row["output_file"]) if row["output_file"] else None
                    if not output or not output.is_file() or output.stat().st_size == 0:
                        self.connection.execute(
                            "UPDATE jobs SET status='pending',output_file=NULL,updated_at=? WHERE sample_id=?",
                            (stamp, prompt.sample_id),
                        )
                elif row["status"] not in ("creating", "submitted", "running"):
                    recovered = recoverable_output(output_dir, prompt.sample_id)
                    if recovered:
                        self.connection.execute(
                            """UPDATE jobs SET status='completed',output_file=?,error_type=NULL,
                               error_message=NULL,updated_at=? WHERE sample_id=?""",
                            (str(recovered.resolve()), stamp, prompt.sample_id),
                        )

    def counts(self, selected_ids: set[str] | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.connection.execute("SELECT sample_id,status FROM jobs"):
            if selected_ids is not None and row["sample_id"] not in selected_ids:
                continue
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return counts

    def pending(self, selected_ids: set[str], limit: int, max_attempts: int) -> list[sqlite3.Row]:
        rows = self.connection.execute(
            "SELECT * FROM jobs WHERE status IN ('pending','failed') AND attempt<? ORDER BY rowid",
            (max_attempts,),
        )
        result: list[sqlite3.Row] = []
        for row in rows:
            if row["sample_id"] in selected_ids:
                result.append(row)
                if len(result) == limit:
                    break
        return result

    def active_batches(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM batches WHERE status IN ('submitted','running') ORDER BY id"
        ).fetchall()

    def creating_batches(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM batches WHERE status='creating' ORDER BY id"
        ).fetchall()

    def create_placeholder(self, rows: list[sqlite3.Row], gcs_root: str) -> sqlite3.Row:
        ids = [row["sample_id"] for row in rows]
        digest = hashlib.sha256("\0".join(ids).encode()).hexdigest()[:12]
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        display_name = f"nano-banana-vertex-{timestamp}-{digest}"
        stamp = utc_now()
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO batches
                   (display_name,status,sample_ids,request_count,pending_count,created_at,updated_at)
                   VALUES (?,'creating',?,?,?,?,?)""",
                (display_name, json.dumps(ids), len(ids), len(ids), stamp, stamp),
            )
            batch_id = int(cursor.lastrowid)
            input_uri = join_gs(gcs_root, "input", f"{display_name}.jsonl")
            output_prefix = join_gs(gcs_root, "output", display_name)
            self.connection.execute(
                "UPDATE batches SET gcs_input_uri=?,gcs_output_prefix=? WHERE id=?",
                (input_uri, output_prefix, batch_id),
            )
            self.connection.executemany(
                """UPDATE jobs SET status='creating',batch_id=?,attempt=attempt+1,
                   error_type=NULL,error_message=NULL,updated_at=? WHERE sample_id=?""",
                [(batch_id, stamp, sample_id) for sample_id in ids],
            )
        return self.connection.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()

    def submitted(self, batch_id: int, job_name: str) -> None:
        stamp = utc_now()
        with self.connection:
            self.connection.execute(
                "UPDATE batches SET job_name=?,status='submitted',updated_at=? WHERE id=?",
                (job_name, stamp, batch_id),
            )
            self.connection.execute(
                "UPDATE jobs SET status='submitted',updated_at=? WHERE batch_id=?",
                (stamp, batch_id),
            )

    def submit_failed(self, batch_id: int, message: str) -> None:
        stamp = utc_now()
        with self.connection:
            self.connection.execute(
                "UPDATE batches SET status='failed',error_message=?,updated_at=? WHERE id=?",
                (message[:12000], stamp, batch_id),
            )
            self.connection.execute(
                """UPDATE jobs SET status='failed',error_type='batch_submit',error_message=?,
                   batch_id=NULL,updated_at=? WHERE batch_id=?""",
                (message[:12000], stamp, batch_id),
            )

    def progress(
        self,
        batch_id: int,
        state: str,
        stats: dict[str, int],
        result_uri: str | None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE batches SET status='running',api_state=?,successful_count=?,
                   failed_count=?,pending_count=?,gcs_result_uri=COALESCE(?,gcs_result_uri),updated_at=?
                   WHERE id=?""",
                (
                    state,
                    stats["successful"],
                    stats["failed"],
                    stats["pending"],
                    result_uri,
                    utc_now(),
                    batch_id,
                ),
            )
            self.connection.execute(
                "UPDATE jobs SET status='running',updated_at=? WHERE batch_id=? AND status IN ('submitted','creating')",
                (utc_now(), batch_id),
            )

    def finish_batch(self, batch_id: int, results: dict[str, tuple[str, str | None]]) -> None:
        stamp = utc_now()
        with self.connection:
            for sample_id, (status, value) in results.items():
                if status == "completed":
                    self.connection.execute(
                        """UPDATE jobs SET status='completed',output_file=?,batch_id=NULL,
                           error_type=NULL,error_message=NULL,updated_at=? WHERE sample_id=?""",
                        (value, stamp, sample_id),
                    )
                else:
                    self.connection.execute(
                        """UPDATE jobs SET status='failed',batch_id=NULL,error_type='request_failed',
                           error_message=?,updated_at=? WHERE sample_id=?""",
                        ((value or "Vertex result contained no image")[:12000], stamp, sample_id),
                    )
            self.connection.execute(
                "UPDATE batches SET status='completed',api_state='JOB_STATE_SUCCEEDED',updated_at=? WHERE id=?",
                (stamp, batch_id),
            )

    def fail_batch(self, batch_id: int, state: str, message: str) -> None:
        stamp = utc_now()
        with self.connection:
            self.connection.execute(
                "UPDATE batches SET status='failed',api_state=?,error_message=?,updated_at=? WHERE id=?",
                (state, message[:12000], stamp, batch_id),
            )
            self.connection.execute(
                """UPDATE jobs SET status='failed',batch_id=NULL,error_type='batch_failed',
                   error_message=?,updated_at=? WHERE batch_id=? AND status!='completed'""",
                (message[:12000], stamp, batch_id),
            )

    def rows_for_batch(self, batch: sqlite3.Row) -> list[sqlite3.Row]:
        ids = json.loads(batch["sample_ids"])
        by_id = {
            row["sample_id"]: row
            for row in self.connection.execute("SELECT * FROM jobs WHERE batch_id=?", (batch["id"],))
        }
        return [by_id[sample_id] for sample_id in ids if sample_id in by_id]

    def close(self) -> None:
        self.connection.close()


class VertexBatchClient:
    """ADC-authenticated wrapper around Vertex REST and Cloud Storage."""

    def __init__(self, project: str, location: str):
        try:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession
            from google.cloud import storage
        except ImportError as exc:
            raise InputError(
                "Missing dependencies: python3 -m pip install -U google-auth google-cloud-storage"
            ) from exc
        credentials, discovered_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not project:
            project = discovered_project or ""
        if not project:
            raise InputError("Set GOOGLE_CLOUD_PROJECT or pass --project")
        self.project = project
        self.location = location
        self.session = AuthorizedSession(credentials)
        self.storage = storage.Client(project=project, credentials=credentials)
        endpoint_prefix = "" if location == "global" else f"{location}-"
        self.api_root = f"https://{endpoint_prefix}aiplatform.googleapis.com/v1"
        self.collection_url = (
            f"{self.api_root}/projects/{quote(project, safe='')}/"
            f"locations/{quote(location, safe='')}/batchPredictionJobs"
        )

    @staticmethod
    def _json_response(response: Any, action: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{action} returned HTTP {response.status_code} with non-JSON body") from exc
        if response.status_code >= 400:
            message = (payload.get("error") or {}).get("message") or json.dumps(payload)
            raise RuntimeError(f"{action} failed HTTP {response.status_code}: {message}")
        return payload

    def upload(self, uri: str, data: bytes) -> None:
        bucket_name, object_name = parse_gs_uri(uri)
        if not object_name:
            raise InputError("GCS input URI needs an object name")
        self.storage.bucket(bucket_name).blob(object_name).upload_from_string(
            data, content_type="application/jsonl"
        )

    def submit(self, display_name: str, model: str, input_uri: str, output_prefix: str) -> dict[str, Any]:
        body = {
            "displayName": display_name,
            "model": f"publishers/google/models/{model}",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": [input_uri]},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": output_prefix},
            },
        }
        response = self.session.post(self.collection_url, json=body, timeout=60)
        return self._json_response(response, "create Vertex batch job")

    def get(self, job_name: str) -> dict[str, Any]:
        response = self.session.get(f"{self.api_root}/{job_name}", timeout=60)
        return self._json_response(response, "get Vertex batch job")

    def find_by_display_name(self, display_name: str) -> dict[str, Any] | None:
        page_token: str | None = None
        for _ in range(20):
            params: dict[str, str | int] = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            response = self.session.get(self.collection_url, params=params, timeout=60)
            payload = self._json_response(response, "list Vertex batch jobs")
            for job in payload.get("batchPredictionJobs") or []:
                if job.get("displayName") == display_name:
                    return job
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return None

    def download_jsonl(self, uri: str) -> list[tuple[str, bytes]]:
        bucket_name, prefix = parse_gs_uri(uri)
        objects: list[tuple[str, bytes]] = []
        for blob in self.storage.list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"):
            if blob.name.endswith(".jsonl"):
                objects.append((blob.name, blob.download_as_bytes()))
        if not objects:
            raise RuntimeError(f"No result JSONL found below {uri}")
        return sorted(objects)


def make_vertex_record(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": row["prompt"]}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "candidateCount": 1,
                "imageConfig": {"aspectRatio": row["aspect_ratio"]},
            },
        }
    }


def encode_jsonl(rows: Iterable[sqlite3.Row]) -> bytes:
    return b"".join(
        (json.dumps(make_vertex_record(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def job_progress(job: dict[str, Any], request_count: int) -> tuple[str, dict[str, int], str | None]:
    state = str(job.get("state") or "JOB_STATE_UNSPECIFIED")
    raw = job.get("completionStats") or {}
    successful = int(raw.get("successfulCount") or 0)
    failed = int(raw.get("failedCount") or 0)
    pending = max(0, request_count - successful - failed)
    result_uri = (job.get("outputInfo") or {}).get("gcsOutputDirectory")
    return state, {"successful": successful, "failed": failed, "pending": pending}, result_uri


def request_prompt(record: dict[str, Any]) -> str | None:
    request = record.get("request") or record.get("instance") or {}
    if "request" in request and isinstance(request["request"], dict):
        request = request["request"]
    texts: list[str] = []
    for content in request.get("contents") or []:
        for part in content.get("parts") or []:
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "".join(texts).strip() or None


def response_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("response") or record.get("prediction")
    if isinstance(value, dict) and isinstance(value.get("response"), dict):
        value = value["response"]
    return value if isinstance(value, dict) else None


def extract_image_from_response(response: dict[str, Any]) -> tuple[bytes, str] | None:
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline or not inline.get("data"):
                continue
            try:
                data = base64.b64decode(inline["data"], validate=True)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f"Invalid base64 image in Vertex output: {exc}") from exc
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            return data, mime
    return None


def record_error(record: dict[str, Any]) -> str:
    error = record.get("error") or record.get("status")
    if isinstance(error, dict):
        return str(error.get("message") or json.dumps(error, ensure_ascii=False))
    return str(error or "Vertex output contained no image")


def save_vertex_results(
    rows: list[sqlite3.Row],
    objects: list[tuple[str, bytes]],
    output_dir: Path,
) -> dict[str, tuple[str, str | None]]:
    by_hash: dict[str, deque[sqlite3.Row]] = defaultdict(deque)
    for row in rows:
        by_hash[row["prompt_sha256"]].append(row)
    records: list[dict[str, Any]] = []
    for object_name, content in objects:
        for line_number, raw_line in enumerate(content.splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {object_name}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)

    assignments: list[tuple[sqlite3.Row, dict[str, Any]]] = []
    unmatched: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()
    for record in records:
        text = request_prompt(record)
        candidates = by_hash.get(sha256_text(text)) if text else None
        if candidates:
            row = candidates.popleft()
            assignments.append((row, record))
            assigned_ids.add(row["sample_id"])
        else:
            unmatched.append(record)

    remaining = [row for row in rows if row["sample_id"] not in assigned_ids]
    if unmatched and len(unmatched) == len(remaining):
        assignments.extend(zip(remaining, unmatched))
        assigned_ids.update(row["sample_id"] for row in remaining)

    results: dict[str, tuple[str, str | None]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for row, record in assignments:
        response = response_payload(record)
        image = extract_image_from_response(response) if response else None
        if not image:
            results[row["sample_id"]] = ("failed", record_error(record))
            continue
        data, mime_type = image
        path = output_dir / f"{row['sample_id']}_nano_banana{output_extension(mime_type)}"
        atomic_write(path, data)
        results[row["sample_id"]] = ("completed", str(path.resolve()))
    for row in rows:
        results.setdefault(row["sample_id"], ("failed", "No matching response in Vertex output JSONL"))
    return results


def selected_prompts(all_prompts: list[Prompt], start: int, limit: int | None) -> list[Prompt]:
    if start < 0 or (limit is not None and limit < 1):
        raise InputError("--start must be >= 0 and --limit must be >= 1")
    return all_prompts[start:] if limit is None else all_prompts[start:start + limit]


def build_parser(script_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", type=Path, default=script_dir / "input/t2i_prompt_bank_1500.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--gcs-uri", default=os.environ.get("NANO_BANANA_GCS_URI"), help="gs://bucket/prefix")
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "output/nano_banana_vertex_batch/remote_v2")
    parser.add_argument("--state-dir", type=Path, default=script_dir / "state/nano_banana_vertex_batch_remote_v2")
    parser.add_argument("--log-dir", type=Path, default=script_dir / "logs")
    parser.add_argument("--expected-count", type=int, default=1500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-test", type=int, choices=(1, 2, 3))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-active-batches", type=int, default=2)
    parser.add_argument("--max-batch-attempts", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument("--max-cost-usd", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def print_progress(db: BatchDB, selected_ids: set[str], active: list[sqlite3.Row]) -> None:
    local = db.counts(selected_ids)
    payload = {
        "local_files_completed": local.get("completed", 0),
        "local_failed": local.get("failed", 0),
        "local_waiting_or_active": sum(v for k, v in local.items() if k not in ("completed", "failed")),
        "active_batches": len(active),
        "vertex_successful": sum(row["successful_count"] for row in active),
        "vertex_failed": sum(row["failed_count"] for row in active),
        "vertex_pending": sum(row["pending_count"] for row in active),
    }
    logging.info("progress=%s", json.dumps(payload))
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    args = build_parser(script_dir).parse_args(argv)
    try:
        if not 1 <= args.chunk_size <= 1000:
            raise InputError("--chunk-size must be between 1 and 1000")
        if args.max_active_batches < 1 or args.max_batch_attempts < 1 or args.expected_count < 0:
            raise InputError("active-batches, attempts, and expected-count values are invalid")
        prompts = selected_prompts(
            load_prompts(args.prompts, args.expected_count),
            args.start,
            args.smoke_test or args.limit,
        )
        if not prompts:
            raise InputError("Selected prompt range is empty")
        estimate = len(prompts) * BATCH_PRICE_PER_IMAGE
        if estimate > args.max_cost_usd:
            raise InputError(f"Estimated batch cost ${estimate:.2f} exceeds --max-cost-usd ${args.max_cost_usd:.2f}")
        if args.gcs_uri:
            parse_gs_uri(args.gcs_uri)
        plan = {
            "backend": "vertex-ai",
            "api_domain": "aiplatform.googleapis.com",
            "authentication": "ADC/service-account OAuth",
            "model": args.model,
            "project": args.project,
            "location": args.location,
            "gcs_uri": args.gcs_uri,
            "prompt_count": len(prompts),
            "chunk_size": args.chunk_size,
            "batch_count": (len(prompts) + args.chunk_size - 1) // args.chunk_size,
            "estimated_image_cost_usd": round(estimate, 2),
            "aspect_ratios": {
                ratio: sum(p.aspect_ratio == ratio for p in prompts)
                for ratio in sorted({p.aspect_ratio for p in prompts})
            },
            "output_dir": str(args.output_dir),
            "state_dir": str(args.state_dir),
        }
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if not args.project:
            raise InputError("Set GOOGLE_CLOUD_PROJECT or pass --project")
        if not args.gcs_uri:
            raise InputError("Set NANO_BANANA_GCS_URI or pass --gcs-uri gs://bucket/prefix")
    except InputError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2

    log_path = setup_logging(args.log_dir, args.verbose)
    logging.info("plan=%s", json.dumps(plan, ensure_ascii=False))
    db: BatchDB | None = None
    stop = [False]

    def handle_signal(signum: int, _frame: Any) -> None:
        logging.warning("received signal %s; no new batches will be submitted", signum)
        stop[0] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        vertex = VertexBatchClient(args.project, args.location)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        db = BatchDB(args.state_dir / "jobs.sqlite3")
        db.seed(prompts, args.model, args.output_dir)
        selected_ids = {prompt.sample_id for prompt in prompts}

        for batch in db.creating_batches():
            existing = vertex.find_by_display_name(batch["display_name"])
            if existing and existing.get("name"):
                db.submitted(batch["id"], existing["name"])
                logging.info("recovered submitted Vertex job=%s", existing["name"])
            else:
                db.submit_failed(batch["id"], "No matching Vertex job found while recovering interrupted submission")

        if args.status:
            for batch in db.active_batches():
                try:
                    job = vertex.get(batch["job_name"])
                    state, stats, result_uri = job_progress(job, batch["request_count"])
                    db.progress(batch["id"], state, stats, result_uri)
                except Exception as exc:
                    logging.warning("status poll failed: %s", exc)
            print_progress(db, selected_ids, db.active_batches())
            return 0

        while not stop[0]:
            active = db.active_batches()
            while len(active) < args.max_active_batches:
                rows = db.pending(selected_ids, args.chunk_size, args.max_batch_attempts)
                if not rows:
                    break
                placeholder = db.create_placeholder(rows, args.gcs_uri)
                try:
                    vertex.upload(placeholder["gcs_input_uri"], encode_jsonl(rows))
                    job = vertex.submit(
                        placeholder["display_name"], args.model,
                        placeholder["gcs_input_uri"], placeholder["gcs_output_prefix"],
                    )
                    if not job.get("name"):
                        raise RuntimeError(f"Vertex create response has no job name: {job}")
                    db.submitted(placeholder["id"], job["name"])
                    logging.info(
                        "submitted Vertex job=%s requests=%d input=%s",
                        job["name"], len(rows), placeholder["gcs_input_uri"],
                    )
                except Exception as exc:
                    db.submit_failed(placeholder["id"], str(exc))
                    logging.error("Vertex batch submission failed: %s", exc)
                active = db.active_batches()

            active = db.active_batches()
            if not active:
                if not db.pending(selected_ids, 1, args.max_batch_attempts):
                    break
                time.sleep(min(5, args.poll_interval))
                continue

            for batch in active:
                try:
                    job = vertex.get(batch["job_name"])
                    state, stats, result_uri = job_progress(job, batch["request_count"])
                    db.progress(batch["id"], state, stats, result_uri)
                    logging.info(
                        "job=%s state=%s successful=%d failed=%d pending=%d",
                        batch["job_name"], state, stats["successful"], stats["failed"], stats["pending"],
                    )
                    if state == "JOB_STATE_SUCCEEDED":
                        uri = result_uri or batch["gcs_result_uri"] or batch["gcs_output_prefix"]
                        objects = vertex.download_jsonl(uri)
                        rows = db.rows_for_batch(batch)
                        results = save_vertex_results(rows, objects, args.output_dir)
                        db.finish_batch(batch["id"], results)
                        logging.info(
                            "downloaded Vertex job=%s images=%d",
                            batch["job_name"], sum(status == "completed" for status, _ in results.values()),
                        )
                    elif state in TERMINAL_STATES:
                        error = job.get("error") or state
                        db.fail_batch(batch["id"], state, json.dumps(error, ensure_ascii=False))
                except Exception as exc:
                    logging.warning("Vertex poll/download failed job=%s error=%s", batch["job_name"], exc)

            print_progress(db, selected_ids, db.active_batches())
            if db.active_batches() or db.pending(selected_ids, 1, args.max_batch_attempts):
                time.sleep(args.poll_interval)

        counts = db.counts(selected_ids)
        summary = {
            "generated_at": utc_now(), "backend": "vertex-ai", "model": args.model,
            "project": args.project, "location": args.location, "counts": counts,
            "output_dir": str(args.output_dir.resolve()),
            "state_db": str((args.state_dir / "jobs.sqlite3").resolve()),
            "log": str(log_path.resolve()),
        }
        args.state_dir.mkdir(parents=True, exist_ok=True)
        (args.state_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 130 if stop[0] else (1 if counts.get("failed", 0) else 0)
    except (InputError, OSError, sqlite3.Error) as exc:
        logging.exception("fatal error: %s", exc)
        return 2
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
