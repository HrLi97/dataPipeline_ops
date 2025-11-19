#!/bin/bash

PROJECT_ROOT="/datas/workspace/wangshunyao/dataPipeline_ops"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
PYTHON_MODULE="pipeline.caption"

INPUT_CSV="/datas/workspace/wangshunyao/dataPipeline_ops/tmp/tmp_single_image.csv/video_list.csv"
OUTPUT_JSON="/datas/workspace/wangshunyao/dataPipeline_ops/tmp/result_caption.json"
LOG_DIR="/datas/workspace/wangshunyao/dataPipeline_ops/tmp/ray_log"
MODEL_PATH="/datas/workspace/wangshunyao/dataPipeline_ops/ckps/Qwen2.5-VL-7B-Instruct"

python3 -m "$PYTHON_MODULE" \
  --is_local True \
  --input_csv_path "$INPUT_CSV" \
  --output_csv_path "$OUTPUT_JSON" \
  --ray_log_dir "$LOG_DIR" \
  --model_id "$MODEL_PATH" \
  --device "cuda" \
  --max_new_tokens 128

if [ $? -eq 0 ]; then
    echo "Pipeline 执行完毕。结果已保存至: $OUTPUT_JSON"
else
    echo "Pipeline 执行出错，请检查日志。"
fi