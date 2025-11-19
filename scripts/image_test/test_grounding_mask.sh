#!/bin/bash
PROJECT_ROOT="/datas/workspace/wangshunyao/dataPipeline_ops"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
PYTHON_MODULE="common.image.grounding_mask_op"
python -m "$PYTHON_MODULE"