"""
MinIO 上传算子。
"""
import os
from minio import Minio
from ..base_ops import BaseOps

class MinioUploadOp(BaseOps):
    """
    将切片上传到 MinIO。支持按 item["segments"] 中的 out_path 批量上传。
    init 参数可以传入已构造好的 Minio client,或通过 endpoint/access_key/secret_key 创建。
    """

    def __init__(self, minio_client=None, bucket=None, prefix="", endpoint=None, access_key=None, secret_key=None, secure=True, **kwargs):
        # 如果传入 client，则使用它；否则尝试根据参数创建（需要安装 minio）
        self.client = minio_client
        self.bucket = bucket
        self.prefix = prefix
        if self.client is None and endpoint and access_key and secret_key:
            self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def predict(self, item: dict) -> dict:
        if self.client is None:
            item["minio_upload"] = "skipped: no client configured"
            return item

        segments = item.get("segments", [])
        uploaded = []
        for seg in segments:
            out_path = seg.get("out_path")
            if not out_path or not self.bucket:
                uploaded.append({"out_path": out_path, "uploaded": False, "reason": "no out_path or bucket"})
                continue
            # 目标对象名
            obj_name = os.path.join(self.prefix, os.path.basename(out_path))
            try:
                self.client.fput_object(self.bucket, obj_name, out_path)
                uploaded.append({"out_path": out_path, "uploaded": True, "object_name": obj_name})
            except Exception as e:
                uploaded.append({"out_path": out_path, "uploaded": False, "reason": str(e)})

        item["minio_upload_results"] = uploaded
        return item
