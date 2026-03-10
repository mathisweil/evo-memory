#!/usr/bin/env python3
"""Archive an experiment to Google Cloud Storage.

Uploads the full experiment folder to gs://{bucket}/experiments/{name}/,
then removes everything locally except report.json and plots/.
Sets experiment status to "archived" in manifest.

Usage:
    python scripts/archive_experiment.py experiment_1
    python scripts/archive_experiment.py 1          # shorthand
    python scripts/archive_experiment.py --all       # all finished experiments
"""

import argparse
import json
import os
import shutil
import sys

from google.cloud import storage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
MANIFEST_PATH = os.path.join(EXPERIMENTS_DIR, "manifest.json")

GCS_BUCKET = os.environ.get("GCS_BUCKET", "statistical-nlp")
GCS_PREFIX = "experiments"

# Files/dirs to keep locally after archival
KEEP_LOCAL = {"report.json", "plots"}


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"experiments": {}}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def normalize_name(name):
    if name.isdigit():
        return f"experiment_{name}"
    return name


def upload_directory(bucket, local_dir, gcs_prefix):
    """Upload all files in local_dir to gs://bucket/gcs_prefix/..."""
    uploaded = 0
    total_bytes = 0
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_name = f"{gcs_prefix}/{rel_path}"
            blob = bucket.blob(blob_name)
            file_size = os.path.getsize(local_path)
            print(f"  Uploading: {rel_path} ({file_size / 1024**2:.1f} MB)")
            blob.upload_from_filename(local_path)
            uploaded += 1
            total_bytes += file_size
    return uploaded, total_bytes


def remove_except_keep(experiment_dir):
    """Remove everything in experiment_dir except KEEP_LOCAL entries."""
    removed_bytes = 0
    for entry in os.listdir(experiment_dir):
        if entry in KEEP_LOCAL:
            continue
        entry_path = os.path.join(experiment_dir, entry)
        if os.path.isdir(entry_path):
            for root, dirs, files in os.walk(entry_path):
                for f in files:
                    removed_bytes += os.path.getsize(os.path.join(root, f))
            shutil.rmtree(entry_path)
        else:
            removed_bytes += os.path.getsize(entry_path)
            os.remove(entry_path)
    return removed_bytes


def archive_experiment(experiment_name):
    manifest = load_manifest()

    if experiment_name not in manifest["experiments"]:
        print(f"ERROR: {experiment_name} not found in manifest")
        return False

    status = manifest["experiments"][experiment_name]["status"]
    if status == "archived":
        print(f"SKIP: {experiment_name} already archived")
        return False
    if status != "active":
        print(f"ERROR: {experiment_name} status is '{status}', expected 'active'")
        return False

    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    if not os.path.isdir(experiment_dir):
        print(f"ERROR: {experiment_dir} does not exist")
        return False

    print(f"Archiving {experiment_name} to gs://{GCS_BUCKET}/{GCS_PREFIX}/{experiment_name}/...")

    # Upload to GCS
    client = storage.Client(project=os.environ.get("GCS_PROJECT", "statistical-nlp"))
    bucket = client.bucket(GCS_BUCKET)
    gcs_path = f"{GCS_PREFIX}/{experiment_name}"
    uploaded, total_bytes = upload_directory(bucket, experiment_dir, gcs_path)

    print(f"\nUploaded {uploaded} files ({total_bytes / 1024**3:.2f} GB) "
          f"to gs://{GCS_BUCKET}/{gcs_path}/")

    # Remove local files except report and plots
    freed = remove_except_keep(experiment_dir)
    print(f"Freed {freed / 1024**3:.2f} GB locally (kept report.json + plots/)")

    # Update manifest
    manifest["experiments"][experiment_name]["status"] = "archived"
    manifest["experiments"][experiment_name]["gcs_path"] = f"gs://{GCS_BUCKET}/{gcs_path}"
    save_manifest(manifest)
    print(f"Status updated: {experiment_name} -> archived")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Archive experiment to Google Cloud Storage")
    parser.add_argument("experiment", nargs="?",
                        help="Experiment name or ID (e.g. experiment_1 or 1)")
    parser.add_argument("--all", action="store_true",
                        help="Archive all finished experiments")
    args = parser.parse_args()

    if args.all:
        manifest = load_manifest()
        for name, info in manifest["experiments"].items():
            if info["status"] == "active":
                archive_experiment(name)
                print()
    elif args.experiment:
        name = normalize_name(args.experiment)
        archive_experiment(name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
