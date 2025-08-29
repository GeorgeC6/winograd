## 编译和运行

由于登录节点 (`x86_64`) 和计算节点 (`aarch64`) 的架构不同，推荐在 SLURM 作业中进行编译和运行。

### SLURM 环境运行

```bash
sbatch run.sh
```

运行脚本会自动加载 spack 环境，使用 Makefile 在计算节点上编译项目并运行基准测试。

### OpenBLAS 路径检测优先级

1. 环境变量 `OPENBLAS_DIR`
2. `spack location -i openblas`（动态获取）
3. 系统包管理器安装的 OpenBLAS

### 旧的 gcc 编译方法

```bash
OPENBLAS_DIR=$(spack location -i openblas)
gcc -O3 -march=armv8-a -fopenmp -I${OPENBLAS_DIR}/include main.c naive_conv.c winograd_conv.c -o winograd ${OPENBLAS_DIR}/lib/libopenblas.a -lm -lpthread
```

> 鲲鹏 920 平台的架构是 arm64 所以原有的 spack 下的东西不一定使用，需要根据文档的指引自己建一个 spack 来下东西。

## 已完成

- [x] `sgemm_parallel` 版本转化为 openblas 版本的矩阵乘法
- [x] tile 分块
- [x] `gemm` 采用手写向量化展开

> 配置 openblas 需要一个小时左右

从结果来看，只有第一个改进实现了质的飞跃，第二个和第三个方法对加速的改进有限。

## 未完成/可以改进的思路

1. 从profile分析结果来看，缓存未命中比较多，因此需要提高对缓存的访问性能，一个方法通过内存重排，将原有的通道形式改变成更易于访问的形式，反正就是各种改，tile 的效果并不是很好。

2. 从结果来看，当 C 比较小的时候，优化的效果和原本的效果相当，一个方法是设置当 C 比较小的时候采用其他方法而不采用 winograd 的方法。

3. 其他
