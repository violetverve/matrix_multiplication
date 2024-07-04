# Matrix Multiplication on Orange Pi

## Overview

This project implements matrix-to-matrix multiplication (A, B are 2000x2000 matrices) on an Orange Pi as fast as possible using various methods:
- NumPy
- PyTorch
- OpenBLAS
- C++ with threads and cycles
- OpenCV
- rknn_matmul_api

## Results

The following table summarizes the execution time for each method:

| Method                     | Time (s) |
|----------------------------|----------|
| NumPy                      | 0.823    |
| PyTorch                    | 3.316    |
| OpenBLAS                   | 0.760    |
| C++ with threads and cycles| 2.358    |
| OpenCV                     | 0.970    |
| rknn_matmul_api            | 2.438    |

## CPU Frequency

```shell
$ lscpu | grep "MHz".
CPU max MHz:                     2256.0000
CPU min MHz:                     408.0000

$ cpufreq-info
current CPU frequency is 408 MHz (asserted by call to hardware).
```

## Method Details and Execution Times

### NumPy
```shell
(env311) orangepi@orangepi5plus:~/labs/lab2$ time python mm_numpy.py 
Sum of elements in result matrix C: 245746.38

real	0m0.823s
user	0m3.739s
sys	0m0.117s
```

### PyTorch
```shell
(env311) orangepi@orangepi5plus:~/labs/lab2$ time python mm_torch.py 
Sum of elements in result matrix C: -95635.421875

real	0m3.316s
user	0m4.213s
sys	0m0.244s
```

### OpenBLAS
```shell
g++ -Ofast mm_openblas mm_openblas.cpp -lopenblas
(env311) orangepi@orangepi5plus:~/labs/lab2$ time ./mm_openblas 
Sum of elements in result matrix C: 2.03472e+09

real	0m0.760s
user	0m2.213s
sys	0m1.430s
```

### C++ with threads and cycles
```shell
g++ -std=c++11 -pthread -O3 mm_cycles.cpp -o mm_cycles
(env311) orangepi@orangepi5plus:~/labs/lab2$ time ./mm_cycles 
Sum of result matrix elements: 2.03472e+09

real	0m2.358s
user	0m7.857s
sys	0m0.047s

```

### OpenCV
```shell
(env311) orangepi@orangepi5plus:~/labs/lab2$ time python mm_opencv.py 
Sum of elements in result matrix C: 1999083100.0

real	0m0.970s
user	0m1.468s
sys	0m0.114s
```

### rknn_matmul_api
```shell
g++ -std=c++11 -O3 -I/home/orangepi/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include/ mm_testrknn.cpp -o mm_testrknn -L/rknpu2/runtime/Linux/librknn_api/aarch64/ -lrknnrt

(env311) orangepi@orangepi5plus:~/labs/lab2$ time ./mm_testrknn 
Sum of elements in result matrix C: 38751368129244299264.000000

real	0m2.617s
user	0m0.662s
sys	0m0.152s
```

## Script Generation

The scripts were mainly generated with ChatGPT using similar prompts:

> Write a [programming language: python/cpp] script with [method] for the fastest optimized matrix multiplication of float32 2000x2000; print the sum of the result.

Example prompt:

> Write a python script with numpy for the fastest optimized matrix multiplication of float32 2000x2000; print the sum of the result.

The generated scripts were edited if necessary. For lesser-known methods like rknn_matmul_api, headers and/or documentation were attached. Several iterations with optimization prompts, such as matrix transposition for the C++ with threads approach, were applied. Some edits were necessary for resolving compilation and execution errors.


## Conclusions
- OpenBLAS proved to be the fastest method for matrix multiplication on Orange Pi, closely followed by Numpy.
- C++ with threads and cycles showed considerable execution time, but optimizations could be explored further.
- rknn_matmul_api exhibited unexpected results, indicating the need for further investigation into its performance and potential optimization.
- PyTorch was the slowest method, likely due to its overhead and the nature of its abstraction.
- For practical purposes on Orange Pi, OpenBLAS and Numpy are recommended due to their balance of speed and ease of use.
