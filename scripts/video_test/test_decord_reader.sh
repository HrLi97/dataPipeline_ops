PROJECT_ROOT="/datas/workspace/wangshunyao/dataPipeline_ops" 
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
PYTHON_SCRIPT_PATH="${PROJECT_ROOT}/common/video/decord_reader_op.py"
PYTHON_MODULE="common.video.decord_reader_op"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "错误: Python 脚本未找到于 ${PYTHON_SCRIPT_PATH}"
    exit 1
fi
python -m "$PYTHON_MODULE"