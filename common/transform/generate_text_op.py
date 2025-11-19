# common/transform/generate_op.py
"""
执行 model.generate 并用 processor.batch_decode 得到文本，将输出放到 item['output_text']。
保持生成参数(max_new_tokens=128)。
"""
from ..base_ops import BaseOps
import torch

class GenerateOp(BaseOps):
    def __init__(self, max_new_tokens: int = 128, **kwargs):
        self.max_new_tokens = max_new_tokens

    def predict(self, item: dict) -> dict:
        model = item.get("model")
        processor = item.get("processor")
        inputs = item.get("inputs")
        if model is None or processor is None or inputs is None:
            item['generate_error'] = "model/processor/inputs missing"
            return item

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        try:
            in_ids_batch = inputs.input_ids
            generated_trimmed = [ out_ids[len(in_ids):] for in_ids, out_ids in zip(in_ids_batch, generated_ids) ]
        except Exception:
            generated_trimmed = [g for g in generated_ids]

        # 解码
        output_texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        item['output_text'] = output_texts[0] if isinstance(output_texts, (list,tuple)) else str(output_texts)
        return item

