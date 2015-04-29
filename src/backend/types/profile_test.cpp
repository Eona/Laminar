/*
 * Eona Studio (c) 2015
 */

#include "../../utils/global_utils.h"
#include "../../utils/timer.h"
#include "../eigen/eigen_engine.h"
#include "../vecmat/vecmat_engine.h"
#include "performance_profiler.h"

#include <Eigen/Dense>

#define CL true
#define CUBLAS true

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

int main(int argc, char **argv)
{
#define ELAPSED(instr, factor) \
	t.start(); instr; elap = t.elapsed(); cout << TEST_SIZE*TEST_SIZE*factor / elap << endl;

	CpuTimer t(Timer::Microsec);

	/**************************************
	******* Eigen engine *********
	**************************************/
	int TEST_SIZE = 2000;
	auto m1e = std::make_shared<MatrixXf>(TEST_SIZE, TEST_SIZE);
	auto m2e = std::make_shared<MatrixXf>(TEST_SIZE, TEST_SIZE);
	auto m3e = std::make_shared<MatrixXf>(TEST_SIZE, TEST_SIZE);

	namespace impe = lmn::EigenImpl;

	impe::fill_rand({}, m1e, true);
	impe::fill_rand({}, m2e, true);

	double elap;

	print_title("Eigen throughput");
	ELAPSED(impe::add({m1e, m2e}, m3e, true), 2)
	ELAPSED(impe::sub({m1e, m2e}, m3e, true), 2)
	ELAPSED(impe::mult_t_t({m1e, m2e}, m3e, true), 2)
	ELAPSED(*m3e = m1e->transpose() * *m2e, 2)
	ELAPSED(*m3e = *m1e * m2e->transpose(), 2)
	ELAPSED(impe::negate({m1e}, m3e, true), 1)
	ELAPSED(impe::assign({m1e}, m3e, true), 1)
	ELAPSED(impe::sigmoid({m1e}, m3e, true), 1)
	ELAPSED(impe::sin({m1e}, m3e, true), 1)
	ELAPSED(impe::cos({m1e}, m3e, true), 1)
	ELAPSED(impe::element_mult({m1e, m2e}, m3e, true), 2)

	TEST_SIZE = 1000;
	using Vecmatf = lmn::Vecmatf;
	auto m1v = std::make_shared<Vecmatf>(TEST_SIZE, TEST_SIZE);
	auto m2v = std::make_shared<Vecmatf>(TEST_SIZE, TEST_SIZE);
	auto m3v = std::make_shared<Vecmatf>(TEST_SIZE, TEST_SIZE);

	/**************************************
	******* Vecmat engine *********
	**************************************/
	namespace impv = lmn::VecmatImpl;

	FakeRand::instance_connection().gen_uniform_rand(TEST_SIZE*TEST_SIZE*2, -1, 1, DEBUG_SEED);

	impv::fill_rand({}, m1v, true);
	impv::fill_rand({}, m2v, true);

	print_title("Vecmat throughput");
	ELAPSED(impv::add<1>({m1v, m2v}, m3v, true), 2)
	ELAPSED(impv::sub<1>({m1v, m2v}, m3v, true), 2)
	ELAPSED(*m3v = *m1v * *m2v, 2)
	ELAPSED(*m3v = m1v->transpose() * *m2v, 2)
	ELAPSED(*m3v = *m1v * m2v->transpose(), 2)
	ELAPSED(impv::negate<1>({m1v}, m3v, true), 1)
	ELAPSED(impv::assign<1>({m1v}, m3v, true), 1)
	ELAPSED(impv::sigmoid({m1v}, m3v, true), 1)
	ELAPSED(impv::sin({m1v}, m3v, true), 1)
	ELAPSED(impv::cos({m1v}, m3v, true), 1)
	ELAPSED(impv::element_mult({m1v, m2v}, m3v, true), 2)


	/**************************************
	******* cuBLAS and OpenCL *********
	**************************************/
	MemoryMonitor mm;

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

}


