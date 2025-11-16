"""
根据 item 中的 input_images(字符串形式或 list)和 output_image 构建 placeholders 列表
占位符格式：
[
  {"type": "image", "image": ref_path, "image_id": idx}, ...,
  {"type": "image", "image": orig_img_path}
]
并把 placeholders 写回 item['placeholders'],同时把 input_images 写回 item['input_images_list'](list)
"""
from ..base_ops import BaseOps
import ast

class PlaceholdersOp(BaseOps):
    def __init__(self, **kwargs):
        pass

    def predict(self, item: dict) -> dict:
        # CSV 中原始字段可能是字符串化的 list(你的原脚本用了 ast.literal_eval)
        raw_input_images = item.get("input_images") or item.get("input_images_list") or item.get("body") or item.get("face")
        # 保证 input_images_list 是 list[str]
        if isinstance(raw_input_images, str):
            try:
                input_images = ast.literal_eval(raw_input_images)
            except Exception:
                input_images = [raw_input_images]
        elif isinstance(raw_input_images, list):
            input_images = raw_input_images
        else:
            input_images = []

        orig_img_path = item.get("output_image") or item.get("orig_img") or item.get("orig10_img")
        placeholders = []
        for idx, ref_path in enumerate(input_images, start=1):
            placeholders.append({
                "type": "image",
                "image": ref_path,
                "image_id": idx
            })
        # 最后追加目标图(没有 image_id)
        if orig_img_path:
            placeholders.append({
                "type": "image",
                "image": orig_img_path
            })

        item['placeholders'] = placeholders
        item['input_images_list'] = input_images
        item['orig_img_path'] = orig_img_path
        return item

if __name__ == '__main__':
    import json

    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/image_list.csv"
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        source_image_path = f.readline().strip()

    op = PlaceholdersOp()

    test_item = {
        "input_images": str([source_image_path, "/path/to/another_ref.jpg"]),
        "output_image": "/path/to/target_image.jpg"
    }

    result_item = op.predict(test_item)

    print(f"--- Testing PlaceholdersOp using base path: {source_image_path} ---")
    print("Generated 'placeholders':")
    print(json.dumps(result_item.get('placeholders'), indent=2))
    print("\n--- Test finished. ---")