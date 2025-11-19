"""
把最终的 json_line 写入 output jsonl 文件（追加）。
构造时注入 output_jsonl_path
"""
from ..base_ops import BaseOps
import json
import os

class SaveJsonlOp(BaseOps):
    def __init__(self, output_jsonl_path: str):
        self.output_jsonl_path = output_jsonl_path
        os.makedirs(os.path.dirname(self.output_jsonl_path), exist_ok=True)

    def predict(self, item: dict) -> dict:
        json_line = item.get("json_line")
        if json_line is None:
            return item
        with open(self.output_jsonl_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        item["jsonl_written"] = True
        return item
