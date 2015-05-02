##Performance Profiling

###1. Overview of the computation engines
We have implemented six computation backends and have conducted extensive profiling on their time efficiency. The frameworks are:

1. CUDA
2. cuBLAS
3. OpenCL (GPU)
4. OpenCL (CPU)
5. C++ Eigen
6. VecMat (vector matrix)

CUDA stands for Compute Unified Device Architecture, is a parallel computing platform and programming model created by NVIDIA and implemented by the graphics processing units (GPUs) that they produce. It is a simplistic interface for NVIDIA Graphical Card. At the same time, it provides extensive access to many low-level functions such as event-based profiling and memory profiling. 

cuBLAS is a linear algebra library of CUDA architecture. cuBLAS is essentially a GPU-accelerated version of the complete standard BLAS library that delivers 6x to 17x faster performance than the latest MKL BLAS. 

OpenCL is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors. Although OpenCL is not as easy to use as CUDA in terms of interface, it provides the possibility of our framework running on numerous devices other than computers that have standard Graphics Processing Unit.

Eigen is a high-level C++ library of template headers for linear algebra, matrix and vector operations, numerical solvers and related algorithms. It is heavily optimized, but can only run on a single thread if not configured with multi-processing option.

VecMat stands for vector matrix, which is implemented using std::vector library. We did not implement any optimization algorithms in this framework so it was expected to be the most inefficient backend. It serves as a baseline of our performance profiling.


###2. Individual operation profiling

In this section, we evaluate the time efficiency performance of indivdual mathematical operations on matrix in different computation engines. Here is a list of operations we evaluate in thie section:

1. Element-wise addition/subtraction
2. Element-wise multiplication
3. Element-wise Scaling
4. Element-wise Sigmoid 
5. Element-wise Sin
6. Element-wise Cos
7. Matrix Multiplication (no transpose)
8. Matrix Multiplication (left operand transpose)
9. Matrix Multiplication (right operand transpose)

Operation 1-6 are element-wise operations, so they should have similar timing profile. We will focus on analysing operations 7-9 because matrix multiplications are O(n^3) operation with naive implementation.

In this section, the performance of an operation are evaluated based on data throughput, which is defined as the input data size per unit time. All input data used in the experiment are in single-precision floating-point format.

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/throughput.png)

According to the figure, all element-wise operations are roughly 200 times faster than matrix multiplication operations, which are as expected. 

One noticable result is that OpenCL has better performance than Plain CUDA in most of the element-wise operations. 

We believe this is because of the fact that OpenCL is a lower-level interface whereas CUDA is a much abstracted interface, which can cause some overhead. 

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/performance_mult.png)

In terms of matrix multiplication, there is also a gap between the GPU-based engine and CPU-based engine in terms of data throughput. For example, Plain CUDA and is about 100 times faster than Eigen and VecMat. 

One noticable result is that in CPU-based engines (except Eigen) and OpenCL(GPU) engine, Matrix Multiplication (left operand transpose) is faster than Matrix Multiplication (no transpose), which is also faster than Matrix Multiplication (right side transpose). This is because of the fact that without optimization, the algorithm loops through the matrix and multiply the elements in whichever way the matrix is ordered. 

For example, in no transpose matrix multiplicaiton operation, the stride size for the right operand is 1, but the stride size of the left operand is the number of rows, which makes memory access extremely inefficent. Matrix Multiplication (right side transpose) is the worst because the stride width for both operand are the number of rows.

Plain CUDA has non of these overhead because the kernel loads the matrix tile to shared memory first using individual kernels, and then it computes the data on the shared memory, which incurs no overhead whatsoever. Eigen does not have this overhead either because of its optimization

Finally, cuBLAS's matrix multiplication is clearly suprior than any other implementations because of it's a heavily optimized linear algebra library.
