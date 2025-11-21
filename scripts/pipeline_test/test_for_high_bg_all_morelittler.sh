#!/bin/bash
PROJECT_ROOT="/datas/workspace/wangshunyao/dataPipeline_ops"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_part/lbm/src:${PROJECT_ROOT}/third_part/Grounded_SAM2_opt:${PYTHONPATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_part/lbm/src:${PYTHONPATH}"
PYTHON_MODULE="pipeline.for_high_bg_all_morelittler"

INPUT_JSON="${PROJECT_ROOT}/tmp/test_input_bg.jsonl"
OUTPUT_JSONL="${PROJECT_ROOT}/tmp/result_bg_processed.jsonl"
OUTPUT_IMG_DIR="${PROJECT_ROOT}/tmp/out_images_dir"
RAY_LOG_DIR="${PROJECT_ROOT}/tmp/ray_log"


DET_CHECKPOINT="${PROJECT_ROOT}/third_part/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
DET_CONFIG="${PROJECT_ROOT}/third_part/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py"

mkdir -p "$(dirname "$OUTPUT_JSONL")"
mkdir -p "$OUTPUT_IMG_DIR"
mkdir -p "$RAY_LOG_DIR"

python3 -m "$PYTHON_MODULE" \
  --is_local True \
  --input_json_path "$INPUT_JSON" \
  --output_jsonl_root "$OUTPUT_JSONL" \
  --output_dir_root "$OUTPUT_IMG_DIR" \
  --ray_log_dir "$RAY_LOG_DIR" \
  --det_checkpoint "$DET_CHECKPOINT" \
  --det_config "$DET_CONFIG"

# ================= 结果检查 =================

if [ $? -eq 0 ]; then
    echo "Pipeline 执行完毕。"
    echo "结果已保存至: $OUTPUT_JSONL"
    # 统计生成了多少结果
    if [ -f "$OUTPUT_JSONL" ]; then
        COUNT=$(wc -l < "$OUTPUT_JSONL")
        echo "📊 成功处理并写入条目数: $COUNT"
    else
        echo "⚠️  脚本运行成功但没有生成输出文件 (可能是图片不符合分辨率/人数要求)。"
    fi
else
    echo "Pipeline 执行出错，请检查上方 Python 报错日志。"
    exit 1
fi