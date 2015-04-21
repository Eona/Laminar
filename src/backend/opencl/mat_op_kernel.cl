__kernel void dummy(__global float * d, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		d[idx] = 0;
    }
}

/*Vector addition, C = aA + bB */
__kernel void mat_add_kernel(__global float * C, __global float * A, __global float * B, float a, float b, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		C[idx] = a*A[idx] + b*B[idx];
    }
}

/*Element wise multiplicatipn, C = A .* B */
__kernel void mat_elem_mult_kernel(__global float * C, __global float * A, __global float * B, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		C[idx] = A[idx] * B[idx];
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

/*sigmoid */
__kernel void mat_sigmoid_kernel(__global float * Y, __global float * X, int DATA_SIZE)
{
    int idx = get_global_id(0);
    if (idx < DATA_SIZE) {
		Y[idx] = 1.f - (X[idx] * X[idx]);
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
		Y[idx] = 1.f / (1.f + exp(-X[idx]));
	}
}
