/*
 * Eona Studio (c) 2015
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "utils/global_utils.h"
//#include "utils/laminar_utils.h"
#include "utils/debug_utils.h"
#include "parameter.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

class Optimizer
{
public:
	Optimizer() :
		initGuard("Optimizer")
	{}

	virtual ~Optimizer() {}

	void init_engine(EngineBase::Ptr engine)
	{
		initGuard.assert_before_initialize("init_engine");
		this->engine = engine;
	}

	void initialize()
	{
		this->initialize_impl();
		initGuard.initialize();
	}

	virtual void update(ParamContainer::Ptr param)
	{
		initGuard.assert_after_initialize("update");

		for (int p = 0; p < param->size(); ++p)
		{
			this->update_impl(param->param_value(p), param->param_gradient(p));
		}
	}

	/************************************/
	TYPEDEF_PTR(Optimizer);

	template<typename OptimizerT, typename ...ArgT>
	static Optimizer::Ptr make(ArgT&& ... args)
	{
		LMN_STATIC_ASSERT((std::is_base_of<Optimizer, OptimizerT>::value),
				"make() failed: type parameter must be a subclass of Optimizer");

		return std::static_pointer_cast<Optimizer>(
					std::make_shared<OptimizerT>(
							std::forward<ArgT>(args) ...));
	}

protected:
	InitializeGuard<LearningException> initGuard;
	EngineBase::Ptr engine;

	virtual void initialize_impl() = 0;

	virtual void update_impl(Tensor& paramValue, Tensor& paramGradient) = 0;
};

/**
 * Stochastic gradient descent
 */
class SGD : public Optimizer
{
public:
	SGD(float initLearningRate) :
		Optimizer(),
		initLearningRate(initLearningRate)
	{}

	virtual ~SGD() {}

protected:
	Scalor::Ptr learningRate;

	float initLearningRate;

	virtual void initialize_impl()
	{
		this->learningRate = Scalor::make(engine);
		// upload float to Scalor
		*learningRate = initLearningRate;
	}

	virtual void update_impl(Tensor& paramValue, Tensor& paramGradient)
	{
		paramValue -= *learningRate * paramGradient;
	}
};

#endif /* OPTIMIZER_H_ */
