#!/bin/bash
#SBATCH --job-name=winograd
#SBATCH --output=winograd_%j.out
#SBATCH --error=winograd_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00
#SBATCH --partition=kunpeng

# 设置OpenMP线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 显示环境信息
echo "Starting Winograd convolution benchmark..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OpenMP threads: $OMP_NUM_THREADS"
echo "Working directory: $(pwd)"

# 检查系统工具
echo "System tools check:"
echo "GCC version: $(gcc --version | head -n1)"

# 加载 spack 环境
echo "==============================="
echo "Loading spack environment..."
if [ -f "$HOME/spack/share/spack/setup-env.sh" ]; then
    . $HOME/spack/share/spack/setup-env.sh
    echo "Spack environment loaded from: $HOME/spack/share/spack/setup-env.sh"
elif [ -f "/opt/spack/share/spack/setup-env.sh" ]; then
    . /opt/spack/share/spack/setup-env.sh
    echo "Spack environment loaded from: /opt/spack/share/spack/setup-env.sh"
else
    echo "Warning: spack setup-env.sh not found, checking if spack is already available..."
fi

# 检查 spack 是否可用
if command -v spack >/dev/null 2>&1; then
    echo "Spack command is available"
    spack --version
    
    # 加载 OpenBLAS
    echo "Loading OpenBLAS via spack..."
    spack load openblas
    
    # 获取 OpenBLAS 路径
    OPENBLAS_DIR=$(spack location -i openblas 2>/dev/null)
    if [ -n "$OPENBLAS_DIR" ] && [ -d "$OPENBLAS_DIR" ]; then
        export OPENBLAS_DIR
        echo "Found OpenBLAS at: $OPENBLAS_DIR"
    else
        echo "Error: Could not get valid OpenBLAS path from spack"
        exit 1
    fi
else
    echo "Error: spack command not found after loading environment"
    exit 1
fi

# 验证 OpenBLAS 路径
if [ ! -f "$OPENBLAS_DIR/include/cblas.h" ]; then
    echo "Error: cblas.h not found in $OPENBLAS_DIR/include/"
    exit 1
fi

if [ ! -f "$OPENBLAS_DIR/lib/libopenblas.a" ] && [ ! -f "$OPENBLAS_DIR/lib/libopenblas.so" ]; then
    echo "Error: OpenBLAS library not found in $OPENBLAS_DIR/lib/"
    exit 1
fi

echo "OpenBLAS verification successful!"
echo "Include directory: $OPENBLAS_DIR/include"
echo "Library directory: $OPENBLAS_DIR/lib"

# 清理之前的构建
echo "==============================="
echo "Cleaning previous build..."
make clean

# 编译项目
echo "Compiling project on compute node with Makefile..."
make OPENBLAS_DIR="$OPENBLAS_DIR"

# 确保可执行文件存在
if [ ! -f "./winograd" ]; then
    echo "Error: winograd executable not found after compilation!"
    echo "Checking current directory contents:"
    ls -la
    exit 1
fi

# 确保配置文件存在
if [ ! -f "./inputs/config.txt" ]; then
    echo "Error: config.txt not found in inputs directory!"
    exit 1
fi

# 设置运行时库路径
echo "Setting up runtime library path..."
export LD_LIBRARY_PATH="$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 运行程序
echo "==============================="
echo "Running Winograd convolution with config.txt..."
./winograd inputs/config.txt

echo "==============================="
echo "Benchmark completed!"