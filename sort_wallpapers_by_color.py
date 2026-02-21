#!/usr/bin/env python3
"""
sort_wallpapers_by_color.py

Non-recursive: processes only files directly inside the SOURCE_DIR.
Detects dominant hue using HSV histogram and moves or copies files into DEST_DIR/<color>/.

Defaults are set for your system:
  SOURCE_DIR = /home/azeem/Shadow/Media/Images/Wallpaper
  DEFAULT_DEST = /home/azeem/Shadow/Media/Images/Wallpaper/sorted_by_color
"""
import os
import argparse
import shutil
import csv
from collections import Counter

from PIL import Image, ImageFile
import numpy as np
import colorsys

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------ CONFIG DEFAULTS ------------
DEFAULT_SOURCE = "/home/azeem/Shadow/Media/Images/Wallpaper"
DEFAULT_DEST = os.path.join(DEFAULT_SOURCE, "sorted_by_color")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

# Hue ranges (degrees). Ranges are inclusive of start and exclusive of end
COLOR_RANGES = {
    "red":    [(345, 360), (0, 15)],
    "orange": [(15, 30)],
    "yellow": [(30, 60)],
    "green":  [(60, 170)],
    "cyan":   [(170, 200)],
    "blue":   [(200, 260)],
    "purple": [(260, 300)],
    "pink":   [(300, 345)],
}

DEFAULT_MIN_SAT = 0.15
DEFAULT_MIN_VAL = 0.15
DEFAULT_RESIZE = (50, 50)
NUM_BINS = 36  # 10-degree bins

# ------------ HELPERS ------------
def is_image_file(fname):
    _, ext = os.path.splitext(fname.lower())
    return ext in ALLOWED_EXT

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)

def get_unique_dest_path(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{i}{ext}")
        i += 1
    return candidate

def hue_to_color_name(hue_deg):
    if hue_deg is None:
        return "neutral"
    for name, ranges in COLOR_RANGES.items():
        for start, end in ranges:
            if start <= end:
                if start <= hue_deg < end:
                    return name
            else:
                if hue_deg >= start or hue_deg < end:
                    return name
    return "other"

# ------------ COLOR ANALYSIS ------------
def analyze_image_dominant_hue(path, resize=DEFAULT_RESIZE, min_sat=DEFAULT_MIN_SAT, min_val=DEFAULT_MIN_VAL, bins=NUM_BINS):
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            im = im.convert("RGB")
            im = im.resize(resize, Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception as e:
        return None, f"error_opening:{e}"

    pixels = arr.reshape(-1, 3)
    # convert each pixel rgb->hsv via colorsys
    hsv = np.array([colorsys.rgb_to_hsv(r, g, b) for (r, g, b) in pixels])
    h = hsv[:, 0] * 360.0
    s = hsv[:, 1]
    v = hsv[:, 2]

    mask = (s >= min_sat) & (v >= min_val)
    if mask.sum() == 0:
        mean_v = float(v.mean())
        mean_s = float(s.mean())
        if mean_v < 0.12:
            return None, "too_dark"
        if mean_s < 0.08:
            return None, "low_saturation"
        return None, "no_pixels_after_filter"

    weights = (s[mask] * v[mask]) + 1e-6
    hist, bin_edges = np.histogram(h[mask], bins=bins, range=(0.0, 360.0), weights=weights)
    if hist.sum() == 0:
        return None, "empty_histogram"

    max_idx = int(hist.argmax())
    bin_start = bin_edges[max_idx]
    bin_end = bin_edges[max_idx + 1]
    dominant_hue = float((bin_start + bin_end) / 2.0) % 360.0

    return dominant_hue, f"hist_peak_bin_{max_idx}"

# ------------ MAIN PROCESS ------------
def process_folder(src, dest, move_files=True, dry_run=True, min_sat=DEFAULT_MIN_SAT, min_val=DEFAULT_MIN_VAL, resize=DEFAULT_RESIZE, verbose=True):
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    safe_makedirs(dest)

    entries = [e for e in os.scandir(src) if e.is_file()]
    total = len(entries)
    if verbose:
        print(f"Found {total} files in {src} (non-recursive).")

    log_rows = []
    counts = Counter()
    skipped = 0
    errors = 0

    for entry in entries:
        fname = entry.name
        fpath = entry.path

        if not is_image_file(fname):
            if verbose:
                print(f" SKIP (not image): {fname}")
            skipped += 1
            continue

        hue, reason = analyze_image_dominant_hue(fpath, resize=resize, min_sat=min_sat, min_val=min_val)
        color = hue_to_color_name(hue)

        if hue is None and reason in ("too_dark",):
            color = "dark"
        elif hue is None and reason in ("low_saturation",):
            color = "neutral"

        target_dir = os.path.join(dest, color)
        safe_makedirs(target_dir)
        dest_path = get_unique_dest_path(target_dir, fname)

        action = "DRY-RUN copy" if (dry_run and not move_files) else ("DRY-RUN move" if dry_run and move_files else ("copy" if not move_files else "move"))
        if verbose:
            print(f"[{action}] {fname} -> {color} (hue={hue if hue is not None else 'N/A'}, reason={reason})")

        if not dry_run:
            try:
                if move_files:
                    shutil.move(fpath, dest_path)
                else:
                    shutil.copy2(fpath, dest_path)
            except Exception as e:
                print(f"  ERROR moving/copying {fname}: {e}")
                errors += 1
                log_rows.append((fpath, color, "", "error", str(e)))
                continue

        counts[color] += 1
        log_rows.append((fpath, color, dest_path if not dry_run else dest_path, "ok" if not dry_run else "dry-run", reason))

    log_file = os.path.join(dest, "sort_log.csv")
    try:
        with open(log_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "assigned_color", "destination", "status", "reason"])
            writer.writerows(log_rows)
    except Exception as e:
        print(f"Failed to write log: {e}")

    print()
    print("=== Summary ===")
    print(f"Total files encountered: {total}")
    print(f"Images processed: {sum(counts.values())}")
    print(f"Skipped (non-image): {skipped}")
    print(f"Errors: {errors}")
    print("Assigned counts by color:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"Log written to: {log_file}")
    return counts, log_file

# ------------ CLI ------------
def parse_args():
    p = argparse.ArgumentParser(description="Sort wallpapers into folders by dominant color (non-recursive).")
    p.add_argument("--src", default=DEFAULT_SOURCE, help="Source folder (non-recursive).")
    p.add_argument("--dest", default=DEFAULT_DEST, help="Destination root folder.")
    p.add_argument("--move", action="store_true", help="Move files instead of copy.")
    p.add_argument("--copy", action="store_true", help="Copy files instead of move.")
    p.add_argument("--dry-run", action="store_true", default=True, help="Do not actually move files (default true). Pass --no-dry-run to perform moves.")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Perform actual moves/copies.")
    p.add_argument("--min-sat", type=float, default=DEFAULT_MIN_SAT)
    p.add_argument("--min-val", type=float, default=DEFAULT_MIN_VAL)
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=list(DEFAULT_RESIZE))
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    move_files = args.move and not args.copy
    if not args.move and not args.copy:
        move_files = True

    process_folder(
        src=args.src,
        dest=args.dest,
        move_files=move_files,
        dry_run=args.dry_run,
        min_sat=args.min_sat,
        min_val=args.min_val,
        resize=tuple(args.resize),
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
