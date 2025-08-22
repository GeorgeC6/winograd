## 编译
目前采用 gcc 方法编译，Makefile 文件有点小问题

```bash
OPENBLAS_DIR=/home/hpc101/h3220101406/spack/opt/spack/linux-aarch64/openblas-0.3.30-hgqgx5krbsc53mvhqcsdj4bo4hphy3xk

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
