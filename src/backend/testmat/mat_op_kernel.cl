__kernel void dummy(__global float * d, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		d[idx] = 0;
    }
}

/*Vector addition, C = aA + bB */
__kernel void 
mat_add_kernel(__global float * C,
               __global float * A, 
               __global float * B, 
               float a, float b, 
               int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		C[idx] = a*A[idx] + b*B[idx];
    }
}
/*COLUMN MAJOR matrix multiplication*/
__kernel void
mat_mult_NN_kernel(__global float* C, 
                   __global float* A,
                   __global float* B,
                   int m, int n, int l, int k)
{
	int idx = get_global_id(0); 
   	if (idx >= m * k) return;
	size_t b_col = (idx / m) * n;//start index in B
	size_t a_row = (idx % m); //start index in A
	float sum = 0;
	for (int i = 0; i < n; ++i) { //a column of A
		sum += B[b_col + i] * A[a_row + i * m];
	} 
	C[idx] = sum;	 
}

/*COLUMN MAJOR matrix multiplication*/
__kernel void
mat_mult_NT_kernel(__global float* C, 
                   __global float* A,
                   __global float* B,
                   int m, int n, int l, int k)
{
	int idx = get_global_id(0); 
   	if (idx >= m * l) return;
	size_t b_row = (idx / m); //start index in B
	size_t a_row = (idx % m); //start index in A
	float sum = 0;
	for (int i = 0; i < n; ++i) { //a row of A
		sum += B[b_row + i * l] * A[a_row + i * m];
	} 
	C[idx] = sum;	 
}

/*COLUMN MAJOR matrix multiplication*/
__kernel void
mat_mult_TN_kernel(__global float* C, 
                   __global float* A,
                   __global float* B,
                   int m, int n, int l, int k)
{
	int idx = get_global_id(0); 
   	if (idx >= n * k) return;
	size_t b_col = (idx / n) * l; //start index in B
	size_t a_col = (idx % n) * m; //start index in A
	float sum = 0;
	for (int i = 0; i < m; ++i) { //a column of A
		sum += B[b_col + i] * A[a_col + i];
	} 
	C[idx] = sum;	 
}

/*Element wise multiplicatipn, C = A .* B */
__kernel void mat_elem_mult_kernel(__global float * C, __global float * A, __global float * B, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		C[idx] = A[idx] * B[idx];
    }
}

/*Square loss */
__kernel void mat_square_loss_kernel (__global float * C, __global float * A, __global float * B, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		float diff =  A[idx] - B[idx];
		C[idx] = 0.5f*(diff*diff);
    }
}

/*Vector scale Y = aX */
__kernel void mat_scale_kernel(__global float * Y, __global float * X, float a, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = X[idx] * a;
    }
}

__kernel void mat_fill_kernel(__global float * Y, float a, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = a;
    }
}

/*sigmoid */
__kernel void mat_sigmoid_kernel(__global float * Y, __global float * X, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = 1.f / (1.f + exp(-X[idx]));
	}
}

/*sigmoid gradient */
__kernel void mat_sigmoid_gradient_kernel(__global float * Y, __global float * X, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = X[idx] * (1.f - X[idx]);
	}
}

/*sin */
__kernel void mat_sin_kernel(__global float * Y, __global float * X, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = sin(X[idx]);
	}
}

/*cos */
__kernel void mat_cos_kernel(__global float * Y, __global float * X, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = cos(X[idx]);
	}
}

/*tanh */
__kernel void mat_tanh_kernel(__global float * Y, __global float * X, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = tanh(X[idx]);
	}
}

/*tanh gradient */
__kernel void mat_tanh_gradient_kernel(__global float * Y, __global float * X, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = 1.f - (X[idx] * X[idx]);
	}
}

