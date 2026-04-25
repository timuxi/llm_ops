# Makefile for CUTLASS GEMM Operator
CXX = g++
NVCC = nvcc

# 设置CUTLASS路径
CUTLASS_PATH = /home/xcy/Block-Sparse-Attention/csrc/cutlass

# 编译选项
CXXFLAGS = -std=c++17 -O3 -I$(CUTLASS_PATH)/include -I../interface
NVCCFLAGS = -std=c++17 -O3 -I$(CUTLASS_PATH)/include -I../interface -arch=sm_75 --expt-relaxed-constexpr --use_fast_math

# 目标文件
TEST_TARGET = test_cutlass_gemm
BENCH_TARGET = benchmark_cutlass_gemm

COMMON_SOURCES = ../kernel/cutlass_gemm.cu ../kernel/cutlass_gemm_impl.cu

# 默认目标
all: $(TEST_TARGET) $(BENCH_TARGET)

# 链接测试目标
$(TEST_TARGET): test_cutlass_gemm.cu $(COMMON_SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TEST_TARGET) test_cutlass_gemm.cu $(COMMON_SOURCES)

# 链接基准测试目标
$(BENCH_TARGET): benchmark_cutlass_gemm.cu $(COMMON_SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(BENCH_TARGET) benchmark_cutlass_gemm.cu $(COMMON_SOURCES)

# 清理
clean:
	rm -f $(TEST_TARGET) $(BENCH_TARGET)

# 运行测试
run: $(TEST_TARGET)
	./$(TEST_TARGET)

# 运行基准测试
bench: $(BENCH_TARGET)
	./$(BENCH_TARGET)

.PHONY: all clean run bench