conda create -n quant python=3.12 -y
conda activate quant

python -m pip install -U pip
python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com "nvidia-modelopt[onnx]==0.42.0"
python -m pip install -U --no-cache-dir polygraphy onnx_graphsurgeon onnxslim onnxscript onnxconverter-common PyYAML soundfile
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

#download from https://developer.nvidia.com/tensorrt
mkdir -p /home/usrname/temp
cd /home/usrname/temp
tar -xzvf TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-12.9.tar.gz

export TRT_ROOT=/home/usrname/temp/TensorRT-10.15.1.29
export LD_LIBRARY_PATH=$TRT_ROOT/lib:$LD_LIBRARY_PATH

cd $TRT_ROOT/python

python -m pip install \
  ./tensorrt-*-cp312-none-linux_x86_64.whl \
  ./tensorrt_dispatch-*-cp312-none-linux_x86_64.whl \
  ./tensorrt_lean-*-cp312-none-linux_x86_64.whl