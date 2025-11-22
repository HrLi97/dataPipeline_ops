#!/bin/bash
PROJECT_ROOT="/datas/workspace/wangshunyao/dataPipeline_ops"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export DEEPFACE_HOME=/mnt/cfs/shanhai/lihaoran/project/code/dataPipeline_ops/ckps
PYTHON_MODULE="common.image.face_detect_op"
python -m "$PYTHON_MODULE"