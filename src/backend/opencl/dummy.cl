__kernel void dummy_kernel(__global float * d, int num)
{
    int idx = get_global_id(0);
    if (idx < num) {
		d[idx] = 2.f;
    }
}
