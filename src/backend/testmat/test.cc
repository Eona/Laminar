/*
 * Eona Studio (c)2015
 */

#define CL 1
#define CUBLAS 0
#include <iostream>

#if CL
#include "../opencl/ocl_util.h"
#include "../opencl/opencl_engine.h"
#elif CUBLAS
#include "../cublas/cublas_engine.h"
typedef std::shared_ptr<CudaFloatMat> CudaFloatMatPtr;
#else
#include "../cuda/cuda_engine.h"
typedef std::shared_ptr<CudaFloatMat> CudaFloatMatPtr;
#endif

using namespace std;


int main(int argc, char **argv)
{
	MemoryMonitor mm;
/**###############Correctness test###################**/
#if 0
#if CL
	OpenclEngine engine;
#else
	CudaEngine engine;
#endif
	//create testcases

	float t1[9] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10, 5};
	float t2[9] = {0.1, 6.8, 4.9, 2.0, 1, 4, 5, 9, 4};
	float t3[6] = {1.1, 7.8, 5.9, 3.0, 2, 5};
	float t4[8] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10};
	float t5[8] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10};
	float tl[3] = {0, 1, 2};

//
//
#if CL
	OpenclFloatMatPtr m1 (new OpenclFloatMat(t1, 3, 3, engine.cl));
	OpenclFloatMatPtr m2 (new OpenclFloatMat(t2, 3, 3, engine.cl));
	OpenclFloatMatPtr m3 (new OpenclFloatMat(t3, 2, 3, engine.cl));
	OpenclFloatMatPtr m4 (new OpenclFloatMat(t4, 4, 2, engine.cl));
	OpenclFloatMatPtr m5 (new OpenclFloatMat(t5, 4, 2, engine.cl));
	OpenclFloatMatPtr m_label (new OpenclFloatMat(tl, 1,3, engine.cl));
	OpenclFloatMatPtr lm (new OpenclFloatMat());
	OpenclFloatMatPtr ms (new OpenclFloatMat());
	OpenclFloatMatPtr out(new OpenclFloatMat());
	ms->isScalar = true;
	std::vector<OpenclFloatMatPtr> v, v1, v2, v3, vl ,vs;
#else
	CudaFloatMatPtr m1 (new CudaFloatMat(t1, 3, 3));
	CudaFloatMatPtr m2 (new CudaFloatMat(t2, 3, 3));
	CudaFloatMatPtr m3 (new CudaFloatMat(t3, 2, 3));
	CudaFloatMatPtr m4 (new CudaFloatMat(t4, 4, 2));
	CudaFloatMatPtr m5 (new CudaFloatMat(t5, 4, 2));
	CudaFloatMatPtr m_label (new CudaFloatMat(tl, 1,3));
	CudaFloatMatPtr lm (new CudaFloatMat());
	CudaFloatMatPtr ms (new CudaFloatMat());
	CudaFloatMatPtr out(new CudaFloatMat());
	ms->isScalar = true;
	std::vector<CudaFloatMatPtr> v, v1, v2, v3, vl, vs;
#endif

	v = {m1, m2};
	v1 = {m3, m1};
	v2 = {m1, m3};
	v3 = {m4, m5};
	vl = {m1, m_label};

	engine.sub(v, out, false);
	out->print_matrix("m1 - m2");

	engine.add(v, out, true);
	out->print_matrix("m1 + m2");

	engine.negate(v, out, true);
	out->print_matrix("-m1");

	engine.multNN(v, out, true);
	out->print_matrix("m1 * m2");

	engine.multNN(v1, out, false);
	out->print_matrix("m3 * m1");

	engine.multNT(v2, out, false);
	out->print_matrix("m1 * T(m3)");

	engine.multTN(v3, out, false);
	out->print_matrix("T(m4) * m5");

	engine.assign(v1, out, false);
	out->print_matrix("m3 -> out");

	ms->scalar = 3;
	vs = {ms, m1};
	engine.multST(vs, out, false);
	out->print_matrix("m1*3");

	vs = {m1, ms};
	engine.multTS(vs, out, false);
	out->print_matrix("3*m1");

	engine.sigmoid(v, out, false);
	out->print_matrix("sigmod(m1)");

	engine.sigmoid_gradient(v, out, true);
	out->print_matrix("sigmoid_gradient(m1)");

	engine.sin(v, out, true);
	out->print_matrix("sin(m1)");

	engine.cos(v, out, true);
	out->print_matrix("cos(m1)");

	engine.tanh(v, out, true);
	out->print_matrix("tanh(m1)");

	engine.tanh_gradient(v, out, true);
	out->print_matrix("tanh_gradient(m1)");

	engine.element_mult(v, out, true);
	out->print_matrix("m1 .* m2");

    engine.square_loss(v, lm, true);
    cout<<"loss: "<<lm->scalar<<endl;

    engine.label_entropy_loss(vl, lm, true);
    cout<<"entropy: "<<lm->scalar<<endl;

    engine.label_softmax_entropy_gradient(vl, out, false);
	out->print_matrix("softmax(m1)");
#endif
/**###############Consistency test###################**/

#if 0
#if CL
	OpenclEngine engine;
#else
	CudaEngine engine;
#endif
	//create testcases

//
//
#if CL
	OpenclFloatMatPtr m1 (new OpenclFloatMat());
	OpenclFloatMatPtr m2 (new OpenclFloatMat());
	OpenclFloatMatPtr m3 (new OpenclFloatMat());
	OpenclFloatMatPtr m4 (new OpenclFloatMat());
	OpenclFloatMatPtr m5 (new OpenclFloatMat());
	OpenclFloatMatPtr lm (new OpenclFloatMat());
	OpenclFloatMatPtr out(new OpenclFloatMat());

	std::vector<OpenclFloatMatPtr> v, v1, v2, v3, vl, rv;
#else
	CudaFloatMatPtr m1 (new CudaFloatMat());
	CudaFloatMatPtr m2 (new CudaFloatMat());
	CudaFloatMatPtr m3 (new CudaFloatMat());
	CudaFloatMatPtr m4 (new CudaFloatMat());
	CudaFloatMatPtr m5 (new CudaFloatMat());
	CudaFloatMatPtr lm (new CudaFloatMat());
	CudaFloatMatPtr out(new CudaFloatMat());

	std::vector<CudaFloatMatPtr> v, v1, v2, v3, vl, rv;
#endif
    int dim1 = 100;
    int dim2 = 200;
    int dim3 = 700;
    engine.create(m1, {dim1, dim2});
    engine.create(m2, {dim1, dim2});
    engine.create(m3, {dim2, dim3});
    engine.create(m4, {300, 1});
    engine.create(m5, {300, 70});


	engine.fill_rand(rv, m1, true);
	engine.fill_rand(rv, m2, true);
	engine.fill_rand(rv, m3, true);
	engine.fill_rand(rv, m4, true);
	engine.fill_rand(rv, m5, true);

	v = {m1,m2};
	engine.sub(v, out, false);
	out->print_matrix("m1 - m2");

	engine.add(v, out, true);
	out->print_matrix("m1 + m2");

	engine.negate(v, out, true);
	out->print_matrix("-m1");

	v = {m1, m3};

	engine.multNN(v, out, false);
	out->print_matrix("m1 * m3");

//	engine.multNT(v, out, false);
//	out->print_matrix("m1 * T(m3)");
//
//	engine.multTN(v, out, false);
//	out->print_matrix("T(m4) * m5");

	engine.assign(v, out, false);
	out->print_matrix("m1 -> out");

	engine.sigmoid(v, out, false);
	out->print_matrix("sigmod(m1)");

	engine.sigmoid_gradient(v, out, true);
	out->print_matrix("sigmoid_gradient(m1)");

	engine.sin(v, out, true);
	out->print_matrix("sin(m1)");

	engine.cos(v, out, true);
	out->print_matrix("cos(m1)");

	engine.tanh(v, out, true);
	out->print_matrix("tanh(m1)");

	engine.tanh_gradient(v, out, true);
	out->print_matrix("tanh_gradient(m1)");

	v = {m1,m2};
	engine.element_mult(v, out, true);
	out->print_matrix("m1 .* m2");

//    engine.square_loss(v, lm, true);
//    cout<<"loss: "<<lm->scalar<<endl;
//
//    engine.label_entropy_loss(vl, lm, true);
//    cout<<"entropy: "<<lm->scalar<<endl;
//
//    engine.label_softmax_entropy_gradient(vl, out, false);
//	out->print_matrix("softmax(m1)");
#endif

/**###############Performance test###################**/
#if 1

#if CL
    GlobalTimer<cl_event> gt;
    OpenclEngine engine(&gt);
#else
    GlobalTimer<cudaEvent_t> gt;
#if CUBLAS
    CublasEngine engine(&gt);
#else
    CudaEngine engine(&gt);
#endif
#endif

	vector<int> dim = {1000, 1000};

#if CL
	OpenclFloatMatPtr m1 (new OpenclFloatMat());
	OpenclFloatMatPtr m2 (new OpenclFloatMat());
	OpenclFloatMatPtr m3 (new OpenclFloatMat());
	OpenclFloatMatPtr out(new OpenclFloatMat());
	OpenclFloatMatPtr lm (new OpenclFloatMat());
    std::vector<OpenclFloatMatPtr> v, rv;
#else
    CudaFloatMatPtr m1 (new CudaFloatMat());
    CudaFloatMatPtr m2 (new CudaFloatMat());
    CudaFloatMatPtr m3 (new CudaFloatMat());
    CudaFloatMatPtr out(new CudaFloatMat());
    CudaFloatMatPtr lm (new CudaFloatMat());
    std::vector<CudaFloatMatPtr> v, rv;
#endif

    engine.create(m1, dim);
    engine.create(m2, dim);
    engine.create(m3, dim);
    engine.create(out, dim);

    engine.fill_rand(rv, m1, true);
    engine.fill_rand(rv, m2, true);
    engine.fill_rand(rv, m3, true);
    v = {m1, m2};

    for (int i = 0; i < 10; ++i){
    	mm.log_memory();
        engine.sub(v, out, true);
        engine.add(v, out, true);
        engine.negate(v, out, true);
        engine.multNN(v, out, true);
        engine.multNT(v, out, true);
        engine.multTN(v, out, true);
        engine.assign(v, out, true);
        engine.sigmoid(v, out, true);
        engine.sigmoid_gradient(v, out, true);
        engine.sin(v, out, true);
        engine.cos(v, out, true);
        engine.tanh(v, out, true);
        engine.tanh_gradient(v, out, true);
        engine.element_mult(v, out, true);
        engine.square_loss(v, lm, true);
        cout<<"loss: "<<lm->scalar<<endl;
    }
    mm.print_stats(MemoryMonitor::Microsec, "test");
#if CL
    gt.print_stats(GlobalTimer<cl_event>::Microsec, "test");
#else
    gt.print_stats(GlobalTimer<cudaEvent_t>::Microsec, "test");
#endif

#endif
}
