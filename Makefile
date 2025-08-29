CC = gcc
CFLAGS = -std=c11 -O3 -march=armv8.2-a+simd -fopenmp
TARGET = winograd
LDFLAGS = -L${OPENBLAS_DIR}/lib ${OPENBLAS_DIR}/lib/libopenblas.a -lm -lpthread
INCLUDES = -I${OPENBLAS_DIR}/include
SOURCES = main.c naive_conv.c winograd_conv.c

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean