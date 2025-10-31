"""
Main pipeline: caption.py
- Worker lazy init:在子进程/本地进程首次调用时创建算子实例(ModelLoaderOp、PlaceholdersOp、VisionInputsOp、GenerateOp、SaveOutputOp)
- __call__ 中按顺序调用算子:ModelLoader -> Placeholders -> VisionInputs -> Generate -> Save
"""
import os
import csv
import argparse
import logging
import ray
from accelerate import PartialState

# 从 common 导入算子
from common.transform.model_loader_op import ModelLoaderOp
from common.image.placeholders_op import PlaceholdersOp
from common.transform.vision_inputs_op import VisionInputsOp
from common.transform.generate_text_op import GenerateOp
from common.io.save_output_op import SaveOutputOp

parser = argparse.ArgumentParser(description='caption pipeline (keeps local/ray modes similar to original)')
parser.add_argument('--output_csv_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/opens2v_total_part2_build_caption.json')
parser.add_argument('--input_csv_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/opens2v_total_part2_build.csv')
parser.add_argument('--is_local', type=bool, default=False)
parser.add_argument("--ray_log_dir", type=str, default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/ray_log")
# parser.add_argument('--model_id', default="/home/ubuntu/MyFiles/vllm-qwen/Qwen2.5-VL-32B-Instruct")
parser.add_argument('--model_id', default="/datas/workspace/wangshunyao/dataPipeline_ops/ckps/Qwen2.5-VL-7B-Instruct")
parser.add_argument('--max_new_tokens', default=128, type=int)
parser.add_argument('--device', default="cuda", type=str)
opt = parser.parse_args()

class Worker:
    """
    lazy init 算子实例（在子进程中创建模型避免序列化错误）
    """
    def __init__(self):
        self._initialized = False
        # 算子占位
        self.model_loader = None
        self.placeholders_op = None
        self.vision_inputs_op = None
        self.generate_op = None
        self.save_op = None

    def _lazy_init(self):
        if self._initialized:
            return
        logging.info("Worker lazy init: 构造算子并加载模型(如需GPU在子进程/本地进程中)")
        self.model_loader = ModelLoaderOp(model_id=opt.model_id)
        self.placeholders_op = PlaceholdersOp()
        self.vision_inputs_op = VisionInputsOp(device=opt.device)
        self.generate_op = GenerateOp(max_new_tokens=opt.max_new_tokens)
        self.save_op = SaveOutputOp(output_path=opt.output_csv_path)
        init_item = {}
        self.model_loader.predict(init_item)
        self._initialized = True
        logging.info("Worker lazy init 完成")

    def __call__(self, item):
        # item 来自 CSV 行（dict），保持与你原脚本传入的一致结构
        if not self._initialized:
            self._lazy_init()

        try:
            item = self.model_loader.predict(item)
            item = self.placeholders_op.predict(item)

            question = (
                "You are an expert in image understanding. "
                "I will give you a list of images: first the reference images, including image_id, and then one target image at the end. "
                "Please only analyze the content of the last image and use placeholders to represent the main content. "
                "The meaning of a placeholder is to replace pronouns in the description, such as people, women, men, etc., with <img><|image_id|></img>. "
                "Do not assign any placeholders to the target image. placeholders are strictly within the range of image_id. "
                "The output should be concise accurate, describing only the main content without extra symbols. "
            )
            item['question_text'] = question

            item = self.vision_inputs_op.predict(item)
            if item.get("vision_inputs_error"):
                logging.warning(f"vision inputs error: {item.get('vision_inputs_error')}")
                return item

            item = self.generate_op.predict(item)
            item = self.save_op.predict(item)

        except Exception as e:
            logging.exception(f"Worker exception: {e}")

        return item

if __name__ == '__main__':
    os.makedirs(os.path.dirname(opt.ray_log_dir), exist_ok=True, mode=0o777)
    # local mode
    if opt.is_local:
        samples = list(csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        device = f"cuda:{device_id}"
        logging.info(f"local mode device: {device}")
        pred = Worker()
        with distributed_state.split_between_processes(samples, apply_padding=True) as sample:
            for item in sample:
                pred(item)
    else:
        ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": ":".join([
                        "/home/ubuntu/Grounded-SAM-2",
                        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking",
                    ]),
                    "HF_ENDPOINT": "https://hf-mirror.com",
                    "DEEPFACE_HOME": "/home/ubuntu/MyFiles/haoran/cpk/",
                }
            }
        )
        samples = ray.data.read_csv(opt.input_csv_path)
        predictions = samples.map(
            Worker,
            num_gpus=1,
            concurrency=6,
        )
        predictions.write_csv(opt.ray_log_dir)
