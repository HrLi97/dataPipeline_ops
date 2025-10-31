"""
把最终结果(instruction / input_images / output_image)以 JSON 每行追加到输出文件(opt.output_csv_path)。
item 应包含 input_images_list, orig_img_path, output_text。
"""
from ..base_ops import BaseOps
import json
import os

class SaveOutputOp(BaseOps):
    def __init__(self, output_path: str, **kwargs):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def predict(self, item: dict) -> dict:
        input_images = item.get("input_images_list", [])
        orig_img = item.get("orig_img_path") or item.get("output_image")
        output_text = item.get("output_text", "")
        record = {
            "task_type": "subject_driven",
            "instruction": output_text,
            "input_images": input_images,
            "output_image": orig_img
        }
        # 追加写入(与原脚本行为一致)
        with open(self.output_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
        item['saved_record'] = record
        return item
