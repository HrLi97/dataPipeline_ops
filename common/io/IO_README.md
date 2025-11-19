# dataPipeline_ops/common/IO
### 存放关于io的处理算子
1.  generate_save_path_op - 根据视频及切片信息，生成本地保存路径并自动创建目录。
    -   input: file_path, seg_idx, start_time, duration
    -   output: out_path, save_dir

2.  minio_upload_op - 将本地文件（如视频切片）上传到 MinIO 对象存储。
    -   input: segments (包含各文件 out_path 的列表)
    -   output: minio_upload_results

3.  save_jsonl_op - 将 item 中预先构建好的 json_line 字典数据追加写入到指定的 .jsonl 文件中。
    -   input: json_line
    -   output: jsonl_written (bool)

4.  save_output_op - 将生成任务的最终结果（包含指令文本、参考图路径列表、目标图路径）构建为特定格式的字典，并追加写入到输出文件。
    -   input: input_images_list, orig_img_path (或 output_image), output_text
    -   output: saved_record

5.  save_pairs_op - 将提取的最佳人脸参考图(refs)和筛选后的视频帧(gt)保存为本地图片文件，并将对应的“参考图-目标图”配对路径信息追加写入 JSONL 数据集文件。
    -   input: best_faces, frame_scores, vr (视频读取对象), file_path
    -   output: saved_out_dir, jsonl_path