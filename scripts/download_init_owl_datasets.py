#!/usr/bin/env python3
"""
Download all inIT-OWL Kaggle datasets and identify which to use for PdM training.

Steps:
  1. List all datasets for the inIT-OWL organization via Kaggle API (user=...).
  2. Download each dataset to data/downloads/kaggle/init_owl_<name>/.
  3. Scan downloaded data (CSV columns, file names) and recommend which datasets
     are suitable for predictive maintenance training.

Requires: pip install kaggle, and ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY).

Usage:
  python scripts/download_init_owl_datasets.py
  python scripts/download_init_owl_datasets.py --list-only     # only list, no download
  python scripts/download_init_owl_datasets.py --analyze-only  # only analyze already-downloaded
"""

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE / "data" / "downloads" / "kaggle"
INIT_OWL_BASE = DOWNLOAD_DIR / "init_owl"
MANIFEST_PATH = INIT_OWL_BASE / "manifest.json"
RECOMMENDATION_PATH = BASE / "docs" / "INIT_OWL_TRAINING_RECOMMENDATION.md"

# Possible org slugs (Kaggle URL uses inIT-OWL; API may use init-owl)
INIT_OWL_ORG_SLUGS = ["inIT-OWL", "init-owl", "initowl"]

# Column patterns that suggest PdM suitability
PDM_KEYWORDS = {
    "sensor": ["vibration", "temperature", "torque", "speed", "rpm", "current", "pressure", "sound", "acceleration"],
    "target": ["failure", "fault", "failure_mode", "target", "label", "condition", "health", "rul", "remaining"],
    "time": ["timestamp", "datetime", "time", "date", "cycle", "hour"],
    "asset": ["machine", "asset", "equipment", "bearing", "id", "machine_id", "device"],
}


def list_init_owl_datasets_cli():
    """Fallback: list datasets via kaggle CLI (e.g. kaggle datasets list --user inIT-OWL)."""
    import subprocess
    for slug in INIT_OWL_ORG_SLUGS:
        try:
            r = subprocess.run(
                ["kaggle", "datasets", "list", "--user", slug, "-v"],
                capture_output=True, text=True, timeout=30, cwd=str(BASE),
            )
            if r.returncode != 0:
                continue
            # CSV format: ref, title, ... (first column is owner/dataset-name)
            refs = []
            for line in (r.stdout or "").strip().splitlines():
                if "/" in line and not line.startswith("ref,"):
                    # -v prints CSV; first column is ref
                    parts = line.split(",")
                    if parts and "/" in parts[0]:
                        refs.append(parts[0].strip())
            if refs:
                return refs
        except Exception:
            continue
    return []


def list_init_owl_datasets():
    """Return list of dataset refs (owner/dataset-name) for inIT-OWL org."""
    # Try Python API first
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        for slug in INIT_OWL_ORG_SLUGS:
            try:
                datasets = api.dataset_list(user=slug, page=1)
                if datasets and len(datasets) > 0:
                    out = []
                    for d in datasets:
                        if d is None:
                            continue
                        ref = getattr(d, "ref", None)
                        if not ref and hasattr(d, "owner") and hasattr(d, "slug"):
                            ref = f"{d.owner}/{d.slug}"
                        if ref:
                            out.append(ref)
                    if out:
                        return out
            except Exception as e:
                print(f"  List with user={slug!r}: {e}")
                continue
    except Exception as e:
        print(f"Kaggle API auth failed: {e}. Trying CLI...")

    # Fallback: CLI list
    refs = list_init_owl_datasets_cli()
    if refs:
        return refs
    # Fallback: read from init_owl/datasets.txt (one slug per line) if present
    slugs_file = INIT_OWL_BASE / "datasets.txt"
    if slugs_file.exists():
        refs = [line.strip() for line in slugs_file.read_text().splitlines() if line.strip() and "/" in line]
        if refs:
            print(f"Using {len(refs)} slug(s) from {slugs_file}")
            return refs
    print("Ensure ~/.kaggle/kaggle.json exists or set KAGGLE_USERNAME and KAGGLE_KEY.")
    print("Or create data/downloads/kaggle/init_owl/datasets.txt with one dataset slug per line (e.g. init-owl/dataset-name).")
    return []


def run_kaggle_download_cli(dataset_slug: str, dest_dir: Path) -> bool:
    """Fallback: download via kaggle CLI."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    work_dir = dest_dir.parent
    zip_name = dataset_slug.split("/")[-1] + ".zip"
    zip_path = work_dir / zip_name
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(work_dir)],
            capture_output=True, text=True, timeout=600, cwd=str(BASE),
        )
        if result.returncode != 0:
            print(f"  CLI error: {result.stderr or result.stdout}")
            return False
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
            zip_path.unlink()
        return True
    except Exception as e:
        print(f"  CLI error: {e}")
        return False


def analyze_folder(folder: Path) -> dict:
    """Inspect a downloaded folder: files, CSV columns, PdM-relevance."""
    info = {"path": str(folder), "files": [], "csv_columns": {}, "pdm_score": 0, "pdm_reasons": []}
    if not folder.exists():
        return info

    for f in sorted(folder.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(folder))
            info["files"].append(rel)
            if f.suffix.lower() in (".csv", ".txt") and f.stat().st_size < 5_000_000:
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                        reader = csv.reader(fp)
                        header = next(reader, None)
                        if header:
                            info["csv_columns"][rel] = header
                except Exception:
                    pass

    # Score PdM suitability from column names
    all_cols = []
    for cols in info["csv_columns"].values():
        all_cols.extend(c.lower() for c in cols)
    for kind, keywords in PDM_KEYWORDS.items():
        for kw in keywords:
            if any(kw in c for c in all_cols):
                info["pdm_score"] += 1
                info["pdm_reasons"].append(f"has {kind}-like column (e.g. '{kw}')")
                break
    return info


def main():
    parser = argparse.ArgumentParser(description="Download all inIT-OWL datasets and recommend for PdM training")
    parser.add_argument("--output-dir", "-o", default=str(INIT_OWL_BASE), help="Base dir for init_owl downloads")
    parser.add_argument("--list-only", action="store_true", help="Only list datasets, do not download")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze already-downloaded folders")
    args = parser.parse_args()

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("inIT-OWL datasets: list, download, and PdM training recommendation")
    print("=" * 60)

    # 1. List
    if not args.analyze_only:
        print("\nListing inIT-OWL organization datasets...")
        refs = list_init_owl_datasets()
        if not refs:
            print("No datasets found. Check org slug (inIT-OWL / init-owl) and API credentials.")
            if args.list_only:
                sys.exit(0)
            # Fallback: try a known slug if any appears in docs
            refs = []
        else:
            print(f"Found {len(refs)} dataset(s):")
            for r in refs:
                print(f"  - {r}")
            manifest = {"org_slugs_tried": INIT_OWL_ORG_SLUGS, "datasets": refs}
            with open(base / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Wrote {base / 'manifest.json'}")

        if args.list_only:
            sys.exit(0)

        # 2. Download each
        print("\nDownloading...")
        for ref in refs:
            safe_name = ref.replace("/", "_").replace(" ", "_")
            dest = base / safe_name
            print(f"  {ref} -> {dest}")
            if not run_kaggle_download_cli(ref, dest):
                print(f"    FAILED")
    else:
        refs = []
        if (base / "manifest.json").exists():
            with open(base / "manifest.json") as f:
                refs = json.load(f).get("datasets", [])

    # 3. Analyze and recommend
    print("\nAnalyzing downloaded data for PdM training suitability...")
    results = []
    for path in sorted(base.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            info = analyze_folder(path)
            info["name"] = path.name
            results.append(info)

    # Build recommendation doc
    docs_dir = BASE / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# inIT-OWL Datasets: Training Recommendation for PdM",
        "",
        "Generated by `scripts/download_init_owl_datasets.py`.",
        "",
        "## Summary",
        "",
        "| Dataset | PdM score | Recommendation | Notes |",
        "|---------|-----------|----------------|--------|",
    ]
    if not results:
        lines.append("| *(no datasets in init_owl folder yet)* | — | Run script with Kaggle credentials to download. | — |")
    for r in sorted(results, key=lambda x: -x["pdm_score"]):
        score = r["pdm_score"]
        rec = "**Use for training**" if score >= 2 else ("Consider" if score >= 1 else "Low priority")
        notes = "; ".join(r["pdm_reasons"][:3]) if r["pdm_reasons"] else "—"
        lines.append(f"| {r['name']} | {score} | {rec} | {notes} |")
    lines.extend([
        "",
        "## PdM relevance",
        "",
        "- **Score 2+**: Has sensor-like and target/failure-like (or time/asset) columns → good for failure prediction or RUL.",
        "- **Score 1**: Partial (e.g. only sensors or only labels) → may need joining or preprocessing.",
        "- **Score 0**: No obvious PdM columns in sampled CSVs → inspect manually if needed.",
        "",
        "## Per-dataset details",
        "",
    ])
    for r in sorted(results, key=lambda x: -x["pdm_score"]):
        lines.append(f"### {r['name']}")
        lines.append(f"- Path: `{r['path']}`")
        lines.append(f"- PdM score: {r['pdm_score']} ({', '.join(r['pdm_reasons']) or 'none'})")
        if r["csv_columns"]:
            lines.append("- CSV columns (sample):")
            for file, cols in list(r["csv_columns"].items())[:3]:
                lines.append(f"  - `{file}`: {cols[:12]}{'...' if len(cols) > 12 else ''}")
        lines.append("")

    with open(RECOMMENDATION_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {RECOMMENDATION_PATH}")

    print("\nDone. See docs/INIT_OWL_TRAINING_RECOMMENDATION.md for which datasets to implement for training.")


if __name__ == "__main__":
    main()
