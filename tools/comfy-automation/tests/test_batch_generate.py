import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "batch_generate.py"
SPEC = importlib.util.spec_from_file_location("batch_generate", SCRIPT)
bg = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = bg
SPEC.loader.exec_module(bg)


class FakeComfy:
    def __init__(self, output_dir, fail_first=False):
        self.output_dir = output_dir
        self.fail_first = fail_first
        self.submit_count = 0
        self.entries = {}

    def submit(self, workflow, client_id):
        self.submit_count += 1
        if self.fail_first and self.submit_count == 1:
            raise bg.ComfyHTTPError(400, '{"error":{"type":"prompt_outputs_failed_validation"}}')
        prompt_id = f"fake-{self.submit_count}"
        save_node = next(k for k, v in workflow.items() if v["class_type"] == "SaveImage")
        prefix = workflow[save_node]["inputs"]["filename_prefix"]
        subfolder, stem = prefix.split("/", 1)
        folder = self.output_dir / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        filename = stem + "_00001_.png"
        (folder / filename).write_bytes(b"fake-png")
        self.entries[prompt_id] = {
            "status": {"completed": True, "status_str": "success", "messages": []},
            "outputs": {save_node: {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}},
        }
        return prompt_id

    def history(self, prompt_id):
        return self.entries.get(prompt_id)

    def get(self, path):
        if path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        raise AssertionError(path)


def args(resume=False):
    return argparse.Namespace(
        resume=resume, max_retries=2, retry_delay=0, queue_size=1,
        width=None, height=None, client_id="test", job_timeout=10,
        poll_interval=0,
    )


class BatchGenerateTest(unittest.TestCase):
    def test_complete_retry_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            spec = bg.WorkflowSpec.inspect("flux2", SCRIPT.parents[0] / "workflows/flux2_api.json")
            prompts = [{"sample_id": "sample_1", "prompt_text": "test prompt"}]
            db = bg.StateDB(root / "state/jobs.sqlite3")
            self.addCleanup(db.close)
            db.seed_jobs(prompts, ["flux2"], "salt")
            client = FakeComfy(output, fail_first=True)

            result = bg.run_model(args(), "flux2", prompts, spec, client, db, output, [False])
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(client.submit_count, 2)
            row = db.connection.execute("SELECT * FROM jobs").fetchone()
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["attempt"], 2)
            self.assertTrue(all(Path(p).is_file() for p in json.loads(row["output_files"])))
            error = db.connection.execute("SELECT error_type FROM attempts WHERE status='attempt_failed'").fetchone()
            self.assertEqual(error["error_type"], "node_validation")

            resumed = bg.run_model(args(resume=True), "flux2", prompts, spec, client, db, output, [False])
            self.assertEqual(resumed["skipped"], 1)
            self.assertEqual(client.submit_count, 2)

    def test_prompt_validation_and_workflow_detection(self):
        prompts = bg.validate_prompts(SCRIPT.parents[1] / "data/prompts/t2i_prompt_bank_1500.json")
        self.assertEqual(len(prompts), 1500)
        self.assertEqual(len({p["sample_id"] for p in prompts}), 1500)
        flux = bg.WorkflowSpec.inspect("flux2", SCRIPT.parents[0] / "workflows/flux2_api.json")
        sd = bg.WorkflowSpec.inspect("sd35", SCRIPT.parents[0] / "workflows/sd35_api.json")
        self.assertEqual((flux.prompt_node, flux.prompt_field, flux.seed_node, flux.seed_field), ("76", "value", "75:73", "noise_seed"))
        self.assertEqual((sd.prompt_node, sd.prompt_field, sd.seed_node, sd.seed_field), ("16", "text", "3", "seed"))


if __name__ == "__main__":
    unittest.main()
