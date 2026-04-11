#!/bin/bash
clear
echo "CUTLASS Add Operator - Performance Test"
echo "======================================"

# 编译项目
echo "Compiling..."
make clean && make

if [ $? -eq 0 ]; then
    echo ""
    echo "Compilation successful!"
    echo ""
    echo "Running tests..."
    echo ""
    ./test_cutlass_gemm
else
    echo "Compilation failed!"
    exit 1
fi