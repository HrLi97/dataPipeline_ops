#!/bin/bash
PROJECT_ROOT="/datas/workspace/wangshunyao/dataPipeline_ops"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
PYTHON_MODULE="common.video.scene_segmenter_op"
python -m "$PYTHON_MODULE"