#!/bin/bash

for var in "$@"
do
  case $var in
    --scan_module=*)
        scan_module=$(echo $var |cut -f2 -d=)
    ;;
  esac
done

source /auto-round/.azure-pipelines/scripts/change_color.sh
RESET="echo -en \\E[0m \\n" # close color

log_dir="/auto-round/.azure-pipelines/scripts/codeScan/scanLog"
mkdir -p $log_dir

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r /auto-round/requirements.txt
pip install -r /auto-round/requirements-cpu.txt

echo "[DEBUG] list pipdeptree..."
pip install pipdeptree
pipdeptree

python -m pylint -f json --disable=R,C,W,E0606,E1129 --enable=line-too-long --max-line-length=120 --extension-pkg-whitelist=numpy --ignored-classes=TensorProto,NodeProto \
--ignored-modules=tensorflow,keras,torch,torch.quantization,torch.tensor,torchvision,fairseq,mxnet,onnx,onnxruntime,intel_extension_for_pytorch,intel_extension_for_tensorflow,torchinfo,horovod,transformers,deepspeed,deepspeed.module_inject \
/auto-round/${scan_module} > $log_dir/pylint.json

exit_code=$?

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------" && $RESET
cat $log_dir/pylint.json
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Pylint error details." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Pylint check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
