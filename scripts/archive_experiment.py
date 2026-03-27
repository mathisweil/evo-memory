#!/usr/bin/env python3
"""Archive an experiment.

For GCS-native experiments (--gcs): updates manifest status to "archived"
and optionally prunes intermediate checkpoints (keeps only final).

For local experiments: uploads full experiment folder to GCS, then removes
everything locally except report.json and plots/.

Usage:
    python scripts/archive_experiment.py experiment_1
    python scripts/archive_experiment.py 1          # shorthand
    python scripts/archive_experiment.py --all       # all active experiments
    python scripts/archive_experiment.py --gcs 1     # GCS-native experiment
"""

import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from experiment_utils import (
    load_manifest, save_manifest, normalize_name, EXPERIMENTS_DIR
)

MANIFEST_PATH = os.path.join(EXPERIMENTS_DIR, "manifest.json")

# Files/dirs to keep locally after archival
KEEP_LOCAL = {"report.json", "plots"}


def upload_directory(bucket, local_dir, gcs_prefix):
    """Upload all files in local_dir to gs://bucket/gcs_prefix/..."""
    uploaded = 0
    total_bytes = 0
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path: str = os.path.join(root, fname)
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


def archive_experiment_local(experiment_name):
    """Archive a local experiment by uploading to GCS."""
    from google.cloud import storage

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

    gcs_bucket = os.environ.get("GCS_BUCKET", "statistical-nlp")
    gcs_project = os.environ.get("GCS_PROJECT", "statistical-nlp")

    print(f"Archiving {experiment_name} to gs://{gcs_bucket}/experiments/{experiment_name}/...")

    client = storage.Client(project=gcs_project)
    bucket = client.bucket(gcs_bucket)
    gcs_path = f"experiments/{experiment_name}"
    uploaded, total_bytes = upload_directory(bucket, experiment_dir, gcs_path)

    print(f"\nUploaded {uploaded} files ({total_bytes / 1024**3:.2f} GB) "
          f"to gs://{gcs_bucket}/{gcs_path}/")

    freed = remove_except_keep(experiment_dir)
    print(f"Freed {freed / 1024**3:.2f} GB locally (kept report.json + plots/)")

    manifest["experiments"][experiment_name]["status"] = "archived"
    manifest["experiments"][experiment_name]["gcs_path"] = f"gs://{gcs_bucket}/{gcs_path}"
    save_manifest(manifest)
    print(f"Status updated: {experiment_name} -> archived")
    return True


def archive_experiment_gcs(experiment_name, gcs, prune_checkpoints=True):
    """Archive a GCS-native experiment (data already in GCS)."""

    def updater(manifest):
        if experiment_name not in manifest["experiments"]:
            print(f"ERROR: {experiment_name} not found in manifest")
            return manifest
        exp = manifest["experiments"][experiment_name]
        if exp["status"] == "archived":
            print(f"SKIP: {experiment_name} already archived")
            return manifest
        exp["status"] = "archived"
        return manifest

    gcs.update_manifest(updater)

    # Prune intermediate checkpoints, keep only final
    if prune_checkpoints:
        prefix = f"experiments/{experiment_name}/"
        blobs = gcs.list_blobs(prefix)
        pruned = 0
        for blob in blobs:
            name = blob.name
            # Delete periodic checkpoints (es_checkpoint_iter*.pt) and
            # training state files, keep es_checkpoint_final.pt
            if ("/checkpoints/es_checkpoint_iter" in name
                    or "/state/training_state_iter" in name):
                blob.delete()
                pruned += 1
        if pruned:
            print(f"  Pruned {pruned} intermediate checkpoint/state files")

    print(f"Status updated: {experiment_name} -> archived")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Archive experiment to Google Cloud Storage")
    parser.add_argument("experiment", nargs="?",
                        help="Experiment name or ID (e.g. experiment_1 or 1)")
    parser.add_argument("--all", action="store_true",
                        help="Archive all active experiments")
    parser.add_argument("--gcs", action="store_true",
                        help="Archive GCS-native experiment (no upload needed)")
    args = parser.parse_args()

    gcs = None
    if args.gcs:
        from es_finetuning.gcs import GCSClient
        gcs = GCSClient()

    if args.all:
        if gcs:
            manifest, _ = gcs.load_manifest()
        else:
            manifest = load_manifest()
        for name, info in manifest["experiments"].items():
            if info["status"] == "active":
                if gcs:
                    archive_experiment_gcs(name, gcs)
                else:
                    archive_experiment_local(name)
                print()
    elif args.experiment:
        name = normalize_name(args.experiment)
        if gcs:
            archive_experiment_gcs(name, gcs)
        else:
            archive_experiment_local(name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
