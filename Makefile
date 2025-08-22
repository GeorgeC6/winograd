CC = gcc
CFLAGS = -std=c11 -O3 -march=armv8.2-a+simd -fopenmp
TARGET = winograd
OPENBLAS_DIR = /home/hpc101/h3220101406/spack/opt/spack/linux-aarch64/openblas-0.3.30-hgqgx5krbsc53mvhqcsdj4bo4hphy3xk
LDFLAGS = -L${OPENBLAS_DIR}/lib/libopenblas.a -lm -lpthread
INCLUDES = -I${OPENBLAS_DIR}/include
SOURCES = main.c naive_conv.c winograd_conv.c

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean