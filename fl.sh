# export FA_VER="v2.7.2.post1"
# export WHEEL="flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# wget -O "$WHEEL" "https://github.com/Dao-AILab/flash-attention/releases/download/${FA_VER}/${WHEEL}"

# pip install --no-deps "./$WHEEL"


# 适配 torch 2.1.2+cu121 的 flash-attn 版本配置
export FA_VER="v2.7.2.post1"
# 对应 CUDA 12、torch 2.1、Python 3.10 的 wheel 包（cxx11abi=FALSE 适配主流环境）
export WHEEL="flash_attn-2.7.2.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# 下载对应版本的 wheel 包（指定版本避免下载错误）
wget -O "$WHEEL" "https://github.com/Dao-AILab/flash-attention/releases/download/${FA_VER}/${WHEEL}"

# 安装（--no-deps 避免覆盖现有 torch 依赖）
pip install --no-deps "./$WHEEL"

# 可选：验证安装是否成功
python -c "import flash_attn; print(f'flash-attn 版本: {flash_attn.__version__}'); print('安装成功！')"