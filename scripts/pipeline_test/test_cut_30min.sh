#!/bin/bash
PROJECT_ROOT="/datas/workspace/wangshunyao/dataPipeline_ops"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
PYTHON_MODULE="pipeline.cut_30min"

python3 -m "$PYTHON_MODULE" --is_local True \
  --input_csv_path "/datas/workspace/wangshunyao/Datasets/cartoon test/dataset/batch_3_cut/video_list/video_list.csv" \
  --output_dir "/datas/workspace/wangshunyao/Datasets/cartoon test/dataset/batch_3_cut/pipeline_out" \
  --output_csv "/datas/workspace/wangshunyao/Datasets/cartoon test/dataset/batch_3_cut/pipeline_out/summary.csv" \
  --log_path "/datas/workspace/wangshunyao/Datasets/cartoon test/dataset/batch_3_cut/pipeline_out/run.log"

echo "Pipeline 执行完毕。"