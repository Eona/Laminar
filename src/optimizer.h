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
#include "learning_listener.h"

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

	virtual void update(ParamContainer::Ptr param, LearningState::Ptr state)
	{
		initGuard.assert_after_initialize("update");

		for (int p = 0; p < param->size(); ++p)
		{
			this->update_impl(param->param_value(p), param->param_gradient(p), state);
		}
	}

	/************************************/
	TYPEDEF_PTR(Optimizer);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(Optimizer)

	GEN_DOWN_CAST_STATIC_MEMBER(Optimizer)

protected:
	InitializeGuard<LearningException> initGuard;
	EngineBase::Ptr engine;

	virtual void initialize_impl() = 0;

	virtual void update_impl(
		Tensor& paramValue, Tensor& paramGradient, LearningState::Ptr state) = 0;
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

	virtual void update_impl(
			Tensor& paramValue, Tensor& paramGradient, LearningState::Ptr state)
	{
		paramValue -= *learningRate * paramGradient;
	}
};

#endif /* OPTIMIZER_H_ */
