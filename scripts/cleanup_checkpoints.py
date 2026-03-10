#!/usr/bin/env python3
"""Remove checkpoint files from an active experiment to save disk space.

Sets experiment status to "finished" after cleanup, indicating no further
training is expected.

Usage:
    python scripts/cleanup_checkpoints.py experiment_1
    python scripts/cleanup_checkpoints.py 1          # shorthand
    python scripts/cleanup_checkpoints.py --all       # all active experiments
"""

import argparse
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
MANIFEST_PATH = os.path.join(EXPERIMENTS_DIR, "manifest.json")


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


def cleanup_experiment(experiment_name):
    manifest = load_manifest()

    if experiment_name not in manifest["experiments"]:
        print(f"ERROR: {experiment_name} not found in manifest")
        return False

    status = manifest["experiments"][experiment_name]["status"]
    if status == "finished":
        print(f"SKIP: {experiment_name} already finished (checkpoints already cleaned)")
        return False
    if status != "active":
        print(f"ERROR: {experiment_name} status is '{status}', expected 'active'")
        return False

    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)

    # Find and remove checkpoint directories
    total_freed = 0
    removed_count = 0

    for root, dirs, files in os.walk(experiment_dir):
        if os.path.basename(root) == "checkpoints":
            dir_size = sum(
                os.path.getsize(os.path.join(root, f))
                for f in files
            )
            total_freed += dir_size
            removed_count += len(files)
            shutil.rmtree(root)
            print(f"  Removed: {root} ({dir_size / 1024**2:.1f} MB)")

    freed_gb = total_freed / 1024**3
    print(f"\nFreed {freed_gb:.2f} GB ({removed_count} checkpoint files)")

    # Update manifest
    manifest["experiments"][experiment_name]["status"] = "finished"
    save_manifest(manifest)
    print(f"Status updated: {experiment_name} -> finished")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove checkpoints from finished experiments")
    parser.add_argument("experiment", nargs="?",
                        help="Experiment name or ID (e.g. experiment_1 or 1)")
    parser.add_argument("--all", action="store_true",
                        help="Clean up all finished experiments")
    args = parser.parse_args()

    if args.all:
        manifest = load_manifest()
        for name, info in manifest["experiments"].items():
            if info["status"] == "active":
                cleanup_experiment(name)
                print()
    elif args.experiment:
        name = normalize_name(args.experiment)
        cleanup_experiment(name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
