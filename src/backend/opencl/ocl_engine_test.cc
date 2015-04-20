/*
 * Eona Studio (c)2015
 */
#include "ocl_util.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	float * data = new float[100];
    int DATA_SIZE = 100;
    int MEM_SIZE = DATA_SIZE*sizeof(float);


    cl_mem memobj = NULL;
    cl_int ret;


    /* Get Platform and Device Info */
    OclUtilContext cl(true);

    /* Create Memory Buffer */
    memobj = cl.to_device(data, MEM_SIZE);

    /*Build program from source code*/
    cl.build_program("./dummy.cl", "dummy_prog");
    /* Set OpenCL Kernel Parameters */
    cl.register_kernel("dummy_kernel", "dummy_prog");
    /* Set OpenCL Kernel Parameters */
    cl.setup_kernel("dummy_kernel", 0, sizeof(cl_mem), &memobj);
    cl.setup_kernel("dummy_kernel", 1, sizeof(int), &DATA_SIZE);

    /* Execute OpenCL Kernel */
    size_t local_ws = 100;    // Number of work-items per work-group

    size_t global_ws = ceil(double(DATA_SIZE)/double(local_ws))*local_ws;    // Total number of work-items
    cout<<global_ws<<endl;
    cl.exec_kernel("dummy_kernel", global_ws, local_ws);

    /* Copy results from the memory buffer */
    float* out = new float[MEM_SIZE];
    cl.to_host(out, memobj, MEM_SIZE);

	for (int i = 0; i < DATA_SIZE; ++i) {
		cout<<out[i];
	}

    /* Finalization */

    ret = clReleaseMemObject(memobj);


    delete[] out;
    delete[] data;

    return 0;
	//create testcases
//	float t1[9] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10, 5};
//	float t2[9] = {0.1, 6.8, 4.9, 2.0, 1, 4, 5, 9, 4};
//	float t3[6] = {1.1, 7.8, 5.9, 3.0, 2, 5};
//
//
//	CudaFloatMatPtr m1 (new CudaFloatMat(t1, 3, 3));
//	CudaFloatMatPtr m2 (new CudaFloatMat(t2, 3, 3));
//	CudaFloatMatPtr m3 (new CudaFloatMat(t3, 2, 3));
//	CudaFloatMatPtr out(new CudaFloatMat());
//    m1->print_matrix("m1");
//    m2->print_matrix("m2");
//    m3->print_matrix("m3");
//
//	std::vector<CudaFloatMatPtr> v;
//	v.push_back(m1);
//	v.push_back(m2);
//
//	std::vector<CudaFloatMatPtr> v1;
//	v1.push_back(m3);
//	v1.push_back(m1);
//
//	lmn::CudaImpl::add<0>(v, out, false);
//	out->print_matrix("m1 + m2");
//
//	lmn::CudaImpl::sub<0>(v, out, true);
//	out->print_matrix("m1 - m2");
//
//	lmn::CudaImpl::negate<0>(v, out, true);
//	out->print_matrix("-m1");
//
//	lmn::CudaImpl::mult<0, 0>(v, out, true);
//	out->print_matrix("m1 * m2");
//
//	lmn::CudaImpl::mult<0, 0>(v1, out, false);
//	out->print_matrix("m3 * m1");
//
//	lmn::CudaImpl::assign<0>(v1, out, true);
//	out->print_matrix("m3 -> out");
//
//	lmn::CudaImpl::sigmoid(v, out, false);
//	out->print_matrix("sigmod(m1)");
//
//	lmn::CudaImpl::sigmoid_gradient(v, out, true);
//	out->print_matrix("sigmoid_gradient(m1)");
//
//	lmn::CudaImpl::sin(v, out, true);
//	out->print_matrix("sin(m1)");
//
//	lmn::CudaImpl::cos(v, out, true);
//	out->print_matrix("cos(m1)");
//
//	lmn::CudaImpl::tanh(v, out, true);
//	out->print_matrix("tanh(m1)");
//
//	lmn::CudaImpl::tanh_gradient(v, out, true);
//	out->print_matrix("tanh_gradient(m1)");
//
//	lmn::CudaImpl::element_mult(v, out, true);
//	out->print_matrix("m1 .* m2");
//
//	float loss;
//	lmn::CudaImpl::square_loss(v, &loss, true);
//	cout<<"loss: "<<loss<<endl;
//
//	lmn::CudaImpl::fill_rand(v, out, true);
//	out->print_matrix("rand");
//
//	lmn::CudaImpl::debug_fill(v, out, true);
//	out->print_matrix("0.66337");
}
