conda activate quant

export TRT_ROOT="/home/usrname/temp/TensorRT-10.15.1.29"
export PATH="$TRT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}"
trtexec="$TRT_ROOT/bin/trtexec"

base_dir="/home/usrname/data/public_repositories/simplest_stt"
onnx="$base_dir/model_int8.onnx"
eng="$base_dir/model_int8.engine"
fp32="$base_dir/model_fp32.onnx"
calibration_data_path="$base_dir/calib_input.npz"

python quantize/pt2fp32onnx.py
python quantize/make_calib.py

# fp32 onnx -> int8 QDQ onnx
python -m modelopt.onnx.quantization \
    --onnx_path "$fp32" \
    --quantize_mode int8 \
    --output_path "$onnx" \
    --calibration_data_path "$calibration_data_path" \
    --calibration_method entropy \
    --calibration_shapes "input:1x512x80" \
    --high_precision_dtype fp32

# onnx->engine
"$trtexec" \
  --onnx="$onnx" \
  --minShapes="input:1x128x80" \
  --optShapes="input:1x512x80" \
  --maxShapes="input:1x1536x80" \
  --stronglyTyped \
  --memPoolSize=workspace:8192 \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --saveEngine="$eng" \
  --skipInference

# test qps
"$trtexec" \
  --loadEngine="$eng" \
  --minShapes="input:1x128x80" \
  --optShapes="input:1x512x80" \
  --maxShapes="input:1x1536x80" \
  --warmUp=2000 \
  --duration=10 \
  --useCudaGraph \
  --useSpinWait