# pipelines/caption/run.py
"""
Pipeline wrapper for vision captioning using common.VisionCaptionModel.
All imports are at the file top per your requirement.
"""

import os
import sys
import csv
import json
import argparse
import logging

# heavy deps at top
import ray
from accelerate import PartialState
import pandas as pd
import torch

# reuse common utilities
from common.vision_caption import VisionCaptionModel
from common.io_ops import append_jsonl

# default model id (keep your original default)
DEFAULT_MODEL_ID = "/home/ubuntu/MyFiles/vllm-qwen/Qwen2.5-VL-32B-Instruct"


class Worker:
    """
    Worker callable for pipeline.
    - Loads VisionCaptionModel once during construction (heavy).
    - __call__ accepts an item (dict from CSV or simple value), builds placeholders,
      calls model.generate_caption, writes result via common.io_ops.append_jsonl.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, output_jsonl: str = None):
        self.model_id = model_id
        self.output_jsonl = output_jsonl
        # instantiate model (heavy)
        self.vc = VisionCaptionModel(model_id=self.model_id)
        # pre-build the user question used in original code
        self.question = (
            "You are an expert in image understanding. "
            "I will give you a list of images: first the reference images, including image_id, and then one target image at the end. "
            "Please only analyze the content of the last image and use placeholders to represent the main content. "
            "The meaning of a placeholder is to replace pronouns in the description, such as people, women, men, etc., with <img><|image_id|></img>. "
            "Do not assign any placeholders to the target image. placeholders are strictly within the range of image_id. "
            "The output should be concise accurate, describing only the main content without extra symbols. "
            "For example, if there are three images including two reference images and one target image: "
            "Two women walk in the park. The woman on the right is from <img><|image_1|></img>. The woman on the left is from <img><|image_2|></img>."
        )

    def _make_placeholders(self, body_list: list, target_path: str) -> list:
        """
        Build placeholders list exactly as original:
         - for each ref image: {"type":"image","image":path,"image_id": idx}
         - finally append target image entry without image_id
        """
        placeholders = []
        for idx, ref_path in enumerate(body_list, start=1):
            placeholders.append({"type": "image", "image": ref_path, "image_id": idx})
        placeholders.append({"type": "image", "image": target_path})
        return placeholders

    def __call__(self, item):
        """
        item: dict from CSV or other. Expected fields:
            - "input_images" (string) or a python-list-like string (e.g. "['/path/a.jpg','/path/b.jpg']")
            - "output_image": target image path
        The function writes a jsonl line to self.output_jsonl and returns item (possibly augmented).
        """
        # normalize item to dict
        if isinstance(item, str):
            # if ray provides a scalar, treat as file path
            video_item = {"output_image": item}
        else:
            video_item = dict(item)

        try:
            body_list_raw = video_item.get("input_images") or video_item.get("input_image") or ""
            # if it's a string representation of list, use ast-like parsing; keep simple: try json.loads then fallback to eval
            if isinstance(body_list_raw, str):
                try:
                    body_list = json.loads(body_list_raw)
                except Exception:
                    # fallback to python literal eval (dangerous if untrusted)
                    import ast
                    body_list = ast.literal_eval(body_list_raw) if body_list_raw else []
            elif isinstance(body_list_raw, (list, tuple)):
                body_list = list(body_list_raw)
            else:
                body_list = []

            target_path = video_item.get("output_image") or video_item.get("orig_img") or ""

            # build placeholders & call model
            placeholders = self._make_placeholders(body_list, target_path)
            output_text = self.vc.generate_caption(placeholders, self.question, max_new_tokens=128)

            # build output dict (follow original schema)
            output_dict = {
                "task_type": "subject_driven",
                "instruction": output_text,
                "input_images": body_list,
                "output_image": target_path
            }

            # write to jsonl (append)
            if self.output_jsonl:
                append_jsonl(self.output_jsonl, output_dict)

            # attach result to returned item for bookkeeping
            video_item["caption_generated"] = True
            video_item["caption_text"] = output_text

        except Exception as e:
            logging.exception(f"Captioning failed for item {item}: {e}")
            video_item["caption_generated"] = False
            video_item["caption_error"] = str(e)

        return video_item


# keep alias to original name to minimize required changes elsewhere
Woker = Worker


def main():
    parser = argparse.ArgumentParser(description="Caption pipeline using Qwen2.5-VL")
    parser.add_argument("--input_csv_path", default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/opens2v_total_part2_build.csv")
    parser.add_argument("--output_jsonl", default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/opens2v_total_part2_build_caption.json")
    parser.add_argument("--is_local", action="store_true", default=False)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--concurrency", type=int, default=6)
    opt = parser.parse_args()

    os.makedirs(os.path.dirname(opt.output_jsonl), exist_ok=True, mode=0o777)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    worker = Worker(model_id=opt.model_id, output_jsonl=opt.output_jsonl)

    if opt.is_local:
        # local: support PartialState splitting like original
        samples = list(csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        # if you want to use accelerate PartialState to shard between processes, uncomment below
        try:
            distributed_state = PartialState()
            device_id = distributed_state.local_process_index
            device = f"cuda:{device_id}"
            # if you want to set device-aware model loading, consider instantiating per-process instead
        except Exception:
            distributed_state = None

        for item in samples:
            worker(item)
        return

    # distributed mode
    ray.init(address="auto", runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    samples = ray.data.read_csv(opt.input_csv_path)
    predictions = samples.map(Worker(opt.model_id, opt.output_jsonl), num_gpus=1, concurrency=opt.concurrency)
    predictions.write_csv(os.path.dirname(opt.output_jsonl) or "./ray_out")


if __name__ == "__main__":
    main()
