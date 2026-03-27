"""Google Cloud Storage client for experiment management.

Provides a lazy-initialized GCS client with helpers for:
- Manifest read/write with optimistic concurrency (if_generation_match)
- Checkpoint upload/download/cleanup
- File and JSON upload/download
"""

import json
import os
import re
import time

GCS_BUCKET = os.environ.get("GCS_BUCKET", "statistical-nlp")
GCS_PROJECT = os.environ.get("GCS_PROJECT", "statistical-nlp")
GCS_EXPERIMENTS_PREFIX = "experiments"
GCS_PRETRAINED_PREFIX = "NAMM_checkpoints/pretrained"


def _iter_num_from_blob(blob):
    """Extract iteration number from a checkpoint blob name."""
    name = os.path.basename(blob.name)
    match = re.search(r'iter(\d+)', name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse iteration from: {name}")


class GCSClient:
    """Lazy-initialized GCS client with experiment management helpers."""

    def __init__(self, bucket_name=None, project=None):
        self.bucket_name = bucket_name or GCS_BUCKET
        self.project = project or GCS_PROJECT
        self._client = None
        self._bucket = None

    @property
    def client(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client(project=self.project)
        return self._client

    @property
    def bucket(self):
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    # ── Low-level helpers ─────────────────────────────────────────────

    def upload_file(self, local_path, gcs_path):
        """Upload a local file to gs://bucket/gcs_path."""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    def upload_json(self, data, gcs_path):
        """Serialize dict to JSON and upload."""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type="application/json",
        )

    def download_file(self, gcs_path, local_path):
        """Download gs://bucket/gcs_path to local_path."""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)

    def download_json(self, gcs_path):
        """Download and parse a JSON file from GCS."""
        blob = self.bucket.blob(gcs_path)
        return json.loads(blob.download_as_text())

    def blob_exists(self, gcs_path):
        return self.bucket.blob(gcs_path).exists()

    def list_blobs(self, prefix):
        return list(self.bucket.list_blobs(prefix=prefix))

    def delete_blob(self, gcs_path):
        self.bucket.blob(gcs_path).delete()

    # ── Manifest with optimistic concurrency ──────────────────────────

    def load_manifest(self):
        """Load manifest from GCS.

        Returns (manifest_dict, generation). The generation number is
        used for optimistic locking via if_generation_match on writes.
        """
        blob = self.bucket.blob("manifest.json")
        if not blob.exists():
            return {"experiments": {}}, 0
        blob.reload()
        data = json.loads(blob.download_as_text())
        return data, blob.generation

    def save_manifest(self, manifest, generation):
        """Write manifest to GCS with optimistic concurrency.

        Returns True on success, False if generation conflict.
        """
        from google.api_core import exceptions
        blob = self.bucket.blob("manifest.json")
        try:
            blob.upload_from_string(
                json.dumps(manifest, indent=2),
                content_type="application/json",
                if_generation_match=generation,
            )
            return True
        except exceptions.PreconditionFailed:
            return False

    def update_manifest(self, update_fn, max_retries=5):
        """Read-modify-write the manifest with automatic retry.

        update_fn(manifest) should mutate and return the dict.
        Retries on generation conflict up to max_retries times.
        """
        for attempt in range(max_retries):
            manifest, generation = self.load_manifest()
            manifest = update_fn(manifest)
            if self.save_manifest(manifest, generation):
                return manifest
            time.sleep(min(0.1 * (2 ** attempt), 5.0))
        raise RuntimeError(
            f"Failed to update manifest after {max_retries} retries")

    # ── Experiment run paths ──────────────────────────────────────────

    def _run_prefix(self, experiment, method, run_name):
        return f"{GCS_EXPERIMENTS_PREFIX}/{experiment}/{method}/{run_name}"

    # ── Checkpoint operations ─────────────────────────────────────────

    def upload_checkpoint(self, local_path, experiment, method, run_name,
                          filename):
        """Upload a checkpoint .pt file to the right GCS path."""
        gcs_path = (f"{self._run_prefix(experiment, method, run_name)}"
                    f"/checkpoints/{filename}")
        self.upload_file(local_path, gcs_path)

    def upload_training_state(self, state_dict, experiment, method, run_name,
                              filename):
        """Upload a training state JSON to GCS."""
        gcs_path = (f"{self._run_prefix(experiment, method, run_name)}"
                    f"/state/{filename}")
        self.upload_json(state_dict, gcs_path)

    def find_latest_checkpoint(self, experiment, method, run_name):
        """Find the latest periodic checkpoint in GCS.

        Returns (checkpoint_blob_name, iteration) or (None, None).
        """
        prefix = (f"{self._run_prefix(experiment, method, run_name)}"
                  f"/checkpoints/es_checkpoint_iter")
        blobs = self.list_blobs(prefix)
        if not blobs:
            return None, None
        blobs.sort(key=_iter_num_from_blob)
        latest = blobs[-1]
        return latest.name, _iter_num_from_blob(latest)

    def download_latest_checkpoint(self, experiment, method, run_name,
                                   local_dir):
        """Download latest checkpoint + training state.

        Returns (local_checkpoint_path, training_state_dict)
        or (None, None) if no checkpoints exist.
        """
        blob_name, iteration = self.find_latest_checkpoint(
            experiment, method, run_name)
        if blob_name is None:
            return None, None

        # Download checkpoint
        filename = os.path.basename(blob_name)
        local_path = os.path.join(local_dir, "checkpoints", filename)
        self.download_file(blob_name, local_path)

        # Download training state
        state_filename = f"training_state_iter{iteration:03d}.json"
        state_gcs_path = (f"{self._run_prefix(experiment, method, run_name)}"
                          f"/state/{state_filename}")
        training_state = None
        try:
            training_state = self.download_json(state_gcs_path)
        except Exception:
            print(f"  Warning: could not download training state "
                  f"{state_filename}")

        return local_path, training_state

    # ── Pretrained NAMM checkpoints ─────────────────────────────────

    def upload_pretrained(self, local_path):
        """Upload a pretrained NAMM checkpoint to GCS.

        Stored under NAMM_checkpoints/pretrained/<filename>.
        """
        filename = os.path.basename(local_path)
        gcs_path = f"{GCS_PRETRAINED_PREFIX}/{filename}"
        self.upload_file(local_path, gcs_path)
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"  Uploaded: gs://{self.bucket_name}/{gcs_path} ({size_mb:.1f} MB)")
        return gcs_path

    def download_latest_pretrained(self, local_cache_dir="exp_local/pretrained"):
        """Download the most recently uploaded pretrained NAMM checkpoint.

        Uses file size to skip re-downloading if a cached copy exists.
        Returns the local path to the checkpoint.
        """
        blobs = [b for b in self.list_blobs(GCS_PRETRAINED_PREFIX)
                 if b.name.endswith(".pt")]
        if not blobs:
            raise FileNotFoundError(
                "No pretrained NAMM checkpoints found in "
                f"gs://{self.bucket_name}/{GCS_PRETRAINED_PREFIX}/")

        latest = sorted(blobs, key=lambda b: b.updated)[-1]
        filename = os.path.basename(latest.name)
        local_path = os.path.join(local_cache_dir, filename)

        if os.path.exists(local_path) and os.path.getsize(local_path) == latest.size:
            print(f"  NAMM checkpoint cached: {local_path}")
            return local_path

        size_mb = latest.size / 1024**2
        print(f"  Downloading gs://{self.bucket_name}/{latest.name} "
              f"({size_mb:.1f} MB)...")
        self.download_file(latest.name, local_path)
        print(f"  Saved: {local_path}")
        return local_path

    def list_pretrained(self):
        """List all pretrained NAMM checkpoints in GCS."""
        blobs = [b for b in self.list_blobs(GCS_PRETRAINED_PREFIX)
                 if b.name.endswith(".pt")]
        return sorted(blobs, key=lambda b: b.updated)

    def cleanup_old_checkpoints(self, experiment, method, run_name, keep=2):
        """Delete all but the most recent `keep` periodic checkpoints."""
        prefix = (f"{self._run_prefix(experiment, method, run_name)}"
                  f"/checkpoints/es_checkpoint_iter")
        blobs = self.list_blobs(prefix)
        if len(blobs) <= keep:
            return

        blobs.sort(key=_iter_num_from_blob)
        to_delete = blobs[:-keep]
        for blob in to_delete:
            iteration = _iter_num_from_blob(blob)
            blob.delete()
            # Also delete companion training state
            state_path = (
                f"{self._run_prefix(experiment, method, run_name)}"
                f"/state/training_state_iter{iteration:03d}.json")
            try:
                self.delete_blob(state_path)
            except Exception:
                pass

    # ── Run-level uploads ─────────────────────────────────────────────

    def upload_run_file(self, local_path, experiment, method, run_name,
                        filename):
        """Upload a file (config.json, results.json, etc.) to run dir."""
        gcs_path = (f"{self._run_prefix(experiment, method, run_name)}"
                    f"/{filename}")
        self.upload_file(local_path, gcs_path)

    def upload_run_artifacts(self, log_dir, experiment, method, run_name):
        """Upload all non-checkpoint artifacts from a run directory."""
        import glob
        prefix = self._run_prefix(experiment, method, run_name)
        for filename in ("config.json", "results.json", "examples.json"):
            local_path = os.path.join(log_dir, filename)
            if os.path.exists(local_path):
                self.upload_file(local_path, f"{prefix}/{filename}")

    # ── Report collection from GCS ────────────────────────────────────

    def collect_runs_gcs(self, experiment_name):
        """Collect run results directly from GCS (no local download).

        Only downloads small JSON files (results.json, examples.json).
        Returns list of run dicts compatible with generate_report.
        """
        prefix = f"{GCS_EXPERIMENTS_PREFIX}/{experiment_name}/"
        all_blobs = self.list_blobs(prefix)

        # Find all results.json files
        results_blobs = [b for b in all_blobs
                         if b.name.endswith("/results.json")]
        runs = []
        for blob in results_blobs:
            parts = blob.name.split("/")
            # experiments/exp_N/method/run_name/results.json
            # or experiments/exp_N/method/run_name/eval_sub/results.json
            if len(parts) < 4:
                continue
            method = parts[2]
            if method == "plots":
                continue

            results = json.loads(blob.download_as_text())

            if len(parts) == 5:
                # Standard run
                run_name = parts[3]
            elif len(parts) == 6:
                # Eval sub-directory
                run_name = f"{parts[3]}/{parts[4]}"
            else:
                continue

            # Try to get examples.json
            examples_path = blob.name.replace("results.json", "examples.json")
            examples = None
            examples_blob = self.bucket.blob(examples_path)
            if examples_blob.exists():
                examples = json.loads(examples_blob.download_as_text())

            runs.append({
                "method": method,
                "run_name": run_name,
                "results": results,
                "examples": examples,
                "run_dir": f"gs://{self.bucket_name}/{'/'.join(parts[:-1])}",
            })

        return runs
