#!/usr/bin/env python3
"""Upload or list pretrained NAMM checkpoints in GCS.

Usage:
    python scripts/upload_pretrained.py exp_local/pretrained/namm_pretrained_romain_v2.pt
    python scripts/upload_pretrained.py exp_local/pretrained/*.pt
    python scripts/upload_pretrained.py --list
    python scripts/upload_pretrained.py --latest-path
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Import gcs module directly to avoid pulling in torch via es_finetuning.__init__
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "es_finetuning.gcs", os.path.join(REPO_ROOT, "es_finetuning", "gcs.py"))
_gcs_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gcs_mod)
GCSClient = _gcs_mod.GCSClient


def main():
    parser = argparse.ArgumentParser(
        description="Upload or list pretrained NAMM checkpoints in GCS")
    parser.add_argument("files", nargs="*",
                        help="Local .pt file(s) to upload")
    parser.add_argument("--list", action="store_true",
                        help="List pretrained checkpoints in GCS")
    parser.add_argument(
        "--latest-path",
        action="store_true",
        help="Print a local pretrained checkpoint path, preferring an existing cached .pt file and otherwise downloading the latest from GCS",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(REPO_ROOT, "exp_local", "pretrained"),
        help="Local cache directory used with --latest-path",
    )
    args = parser.parse_args()

    gcs = GCSClient()

    if args.latest_path:
        print(gcs.resolve_pretrained_checkpoint(args.cache_dir))
        return

    if args.list:
        blobs = gcs.list_pretrained()
        if not blobs:
            print("No pretrained checkpoints in GCS.")
            return
        print(f"Pretrained checkpoints in gs://{gcs.bucket_name}/:")
        for blob in blobs:
            size_mb = blob.size / 1024**2
            print(f"  {os.path.basename(blob.name):40s} "
                  f"{size_mb:>7.1f} MB  {blob.updated.strftime('%Y-%m-%d %H:%M:%S')}")
        return

    if not args.files:
        parser.print_help()
        return

    for filepath in args.files:
        if not filepath.endswith(".pt"):
            print(f"SKIP: {filepath} (not a .pt file)")
            continue
        if not os.path.isfile(filepath):
            print(f"ERROR: {filepath} not found")
            continue
        size_mb = os.path.getsize(filepath) / 1024**2
        print(f"Uploading {filepath} ({size_mb:.1f} MB)...")
        gcs.upload_pretrained(filepath)


if __name__ == "__main__":
    main()
