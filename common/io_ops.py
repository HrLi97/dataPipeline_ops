# common/io_ops.py
import os
import json
import csv
import cv2
from typing import Dict, Any

def save_image(path: str, img) -> None:
    """Save image (expects BGR for cv2.imwrite)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a JSON line to jsonl file (UTF-8)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fw:
        fw.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    """Append a dict row to a CSV file, write header if new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def generate_output_path(output_root: str, video_file: str, seg_idx: int, start_time: float, duration: float) -> str:
    """
    Build output directory and output file path for a segment.
    Format: <output_root>/<video_name>/<video_name>_seg{seg_idx}_{int(start)}_{int(duration)}.mp4
    """
    base = os.path.basename(video_file)
    name, _ = os.path.splitext(base)
    save_dir = os.path.join(output_root, name)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"{name}_seg{seg_idx}_{int(start_time)}_{int(duration)}.mp4")
