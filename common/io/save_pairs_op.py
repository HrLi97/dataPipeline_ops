"""
将 best_faces(refs)和 selected frames 保存成图片与 jsonl(逐条追加)。
保存 body/ref 图片与 gt 图片，并写入 jsonl 记录 input_images (ref paths) 与 output_image (gt path)。
"""
from ..base_ops import BaseOps
import os
import json
import cv2

class SavePairsOp(BaseOps):
    def __init__(self, output_dir_root, jsonl_path, save_orig=True, **kwargs):
        self.output_dir_root = output_dir_root
        self.jsonl_path = jsonl_path
        self.save_orig = save_orig
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)

    def predict(self, item: dict) -> dict:
        best_faces = item.get("best_faces", [])
        frame_scores = item.get("frame_scores", [])
        vr = item.get("vr")
        name = os.path.splitext(os.path.basename(item.get("file_path","unknown")))[0]
        out_dir = os.path.join(self.output_dir_root, name)
        os.makedirs(out_dir, exist_ok=True)

        # 保存 refs
        ref_paths = []
        for i, entry in enumerate(best_faces):
            ref_path = os.path.join(out_dir, f"ref_{i}.jpg")
            # entry['body'] 是 BGR
            cv2.imwrite(ref_path, entry['body'])
            ref_paths.append(ref_path)

        # 打开 jsonl 追加
        with open(self.jsonl_path, "a", encoding="utf-8") as fw:
            for entry in frame_scores:
                idx = entry['frame_idx']
                matched = entry['matched_refs']
                try:
                    frame = vr.get_batch([idx]).asnumpy()[0]
                except Exception:
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gt_fn = f"gt_{idx}.jpg"
                gt_path = os.path.join(out_dir, gt_fn)
                cv2.imwrite(gt_path, bgr)
                input_images = [ref_paths[r] for r in matched]
                record = {"task_type":"subject-driven","instruction":"","input_images":input_images,"output_image":gt_path}
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")

        item['saved_out_dir'] = out_dir
        item['jsonl_path'] = self.jsonl_path
        return item
