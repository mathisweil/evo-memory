"""Download all experiment artifacts from GCS and WandB.

Creates experiment_artifacts/ with the following structure:
  experiment_artifacts/
  ├── gcs/
  │   ├── M1/                    LoRA-only checkpoint (best_ckpt.pt, config, metrics)
  │   ├── M2_cs1024/             NAMM checkpoint (ckpt.pt, latest.pt, evals)
  │   ├── M2_cs2048/             NAMM checkpoint
  │   ├── M3_cs1024/             LoRA + frozen NAMM checkpoint
  │   └── M3_cs2048/             LoRA + frozen NAMM checkpoint
  └── wandb/
      ├── M1/{kz6vqo2o,x9a4smmf,qfoxxi2m}/    3 LoRA run segments
      ├── M2_cs1024/lenhmfb1/                   NAMM run
      ├── M2_cs2048/{y5fdw0f9,ccflnsds}/        2 NAMM run segments
      ├── M2_cs3072/quc95irz/                   NAMM run (no GCS checkpoint)
      ├── M3_cs1024/ovosogkj/                   LoRA + NAMM run
      ├── M3_cs2048/m4knrhmr/                   LoRA + NAMM run
      └── M3_cs3072/4sgkswa6/                   LoRA + NAMM run (no GCS checkpoint)

Each wandb run directory contains:
  - config.json      Full run configuration
  - summary.json     Final summary metrics
  - history.csv      Full metric history (all logged steps)
  - metadata.json    Run ID, name, state, host, command

Prerequisites:
  pip install wandb google-cloud-storage pandas
  wandb login          # or set WANDB_API_KEY
  gcloud auth login    # or use application default credentials
"""

import json
import os
import sys

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"
GCS_BUCKET = "statistical-nlp"
GCS_PROJECT = "statistical-nlp"

# GCS checkpoint paths (prefix -> local directory name)
GCS_DOWNLOADS = {
    "M2_cs1024": "NAMM_checkpoints/pretrained/namm-5t-cs1024-llama32-1b/",
    "M2_cs2048": "NAMM_checkpoints/pretrained/namm-5t-cs2048-llama32-1b/",
    "M1":        "NAMM_checkpoints/pretrained/lora-m1-5t-llama32-1b/",
    "M3_cs1024": "NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs1024-llama32-1b/",
    "M3_cs2048": "NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs2048-llama32-1b/",
}

# WandB run IDs (subdir -> run ID)
WANDB_RUNS = {
    "M2_cs1024/lenhmfb1": "lenhmfb1",
    "M2_cs2048/y5fdw0f9": "y5fdw0f9",
    "M2_cs2048/ccflnsds":  "ccflnsds",
    "M2_cs3072/quc95irz": "quc95irz",
    "M1/kz6vqo2o":        "kz6vqo2o",
    "M1/x9a4smmf":        "x9a4smmf",
    "M1/qfoxxi2m":        "qfoxxi2m",
    "M3_cs1024/ovosogkj": "ovosogkj",
    "M3_cs2048/m4knrhmr": "m4knrhmr",
    "M3_cs3072/4sgkswa6": "4sgkswa6",
}


def download_gcs(base_dir):
    from google.cloud import storage

    gcs_dir = os.path.join(base_dir, "gcs")
    client = storage.Client(project=GCS_PROJECT)
    bucket = client.bucket(GCS_BUCKET)

    for label, prefix in GCS_DOWNLOADS.items():
        out_dir = os.path.join(gcs_dir, label)
        os.makedirs(out_dir, exist_ok=True)
        blobs = list(client.list_blobs(bucket, prefix=prefix))
        print(f"\n  {label} ({len(blobs)} files)")
        for blob in blobs:
            fname = blob.name.split("/")[-1]
            dest = os.path.join(out_dir, fname)
            if os.path.exists(dest) and os.path.getsize(dest) == blob.size:
                print(f"    {fname} — already exists, skipping")
                continue
            size_mb = blob.size / 1024 / 1024
            print(f"    {fname} ({size_mb:.1f} MB)...")
            blob.download_to_filename(dest)


def download_wandb(base_dir):
    import pandas as pd
    import wandb

    wandb_dir = os.path.join(base_dir, "wandb")
    api = wandb.Api()

    for subdir, rid in WANDB_RUNS.items():
        out_dir = os.path.join(wandb_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)

        # Skip if already downloaded
        if all(os.path.exists(os.path.join(out_dir, f)) for f in ["config.json", "summary.json", "history.csv", "metadata.json"]):
            print(f"\n  {subdir} — already exists, skipping")
            continue

        print(f"\n  {subdir}")
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")

        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(r.config, f, indent=2, default=str)

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(dict(r.summary), f, indent=2, default=str)

        h = r.history(pandas=True, samples=100000)
        h.to_csv(os.path.join(out_dir, "history.csv"), index=False)
        print(f"    history.csv: {len(h)} rows, {len(h.columns)} columns")

        meta = {
            "id": r.id,
            "name": r.name,
            "state": r.state,
            "created_at": r.created_at,
            "tags": r.tags,
            "host": r.metadata.get("host", "unknown") if r.metadata else "unknown",
            "command": r.metadata.get("args", []) if r.metadata else [],
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


def main():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_artifacts")
    os.makedirs(base_dir, exist_ok=True)

    print("=== Downloading GCS checkpoints ===")
    try:
        download_gcs(base_dir)
    except Exception as e:
        print(f"\n  GCS download failed: {e}")
        print("  Run 'gcloud auth application-default login' and retry.")

    print("\n\n=== Downloading WandB run data ===")
    try:
        download_wandb(base_dir)
    except Exception as e:
        print(f"\n  WandB download failed: {e}")
        print("  Run 'wandb login' and retry.")

    # Print summary
    print("\n\n=== Download complete ===")
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    print(f"  Location: {base_dir}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")


if __name__ == "__main__":
    main()
