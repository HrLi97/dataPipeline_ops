# dataPipeline_ops/common/IO
### 存放关于io的处理算子
1.  generate_save_path_op - 根据视频及切片信息，生成本地保存路径并自动创建目录。
    -   input: file_path, seg_idx, start_time, duration
    -   output: out_path, save_dir

2.  minio_upload_op - 将本地文件（如视频切片）上传到 MinIO 对象存储。
    -   input: segments (包含各文件 out_path 的列表)
    -   output: minio_upload_results