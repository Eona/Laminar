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

	void initialize(vector<ParamContainer::Ptr> params)
	{
		this->initialize_impl();
		for (auto param : params)
			for (int p = 0; p < param->size(); ++p)
			{
				this->setup_impl(param->param_value(p));
			}

		initGuard.initialize();
	}

	virtual void update(vector<ParamContainer::Ptr> params, LearningState::Ptr state)
	{
		initGuard.assert_after_initialize("update");

		for (auto param : params)
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

	/**
	 * Set up for momentum/adagrad learning that needs to store history of previous updates
	 * Create a hashmap to keep track of which history belongs to which param tensor
	 * Should NOT modify the value, because they are all zeros anyway (just initialized)
	 */
	virtual void setup_impl(Tensor& paramValue) = 0;

	virtual void update_impl(
		Tensor& paramValue, Tensor& paramGradient, LearningState::Ptr state) = 0;
};

/**
 * Stochastic gradient descent with (decaying) learning rate every epoch
 */
class SimpleSGD : public Optimizer
{
public:
	/**
	 *
	 * @param initLearningRate
	 * @param decay defaults to 1.f (no change over each minibatch)
	 * decay < 1.f to decrease learning, > 1.f to accelerate learning
	 */
	SimpleSGD(float initLearningRate, float decayRate = 1.f) :
		Optimizer(),
		initLearningRate(initLearningRate),
		decayRate(decayRate)
	{}

	virtual ~SimpleSGD() {}

protected:
	Scalar::Ptr learningRate;
	Scalar::Ptr decay;

	float initLearningRate;
	float decayRate;

	virtual void initialize_impl()
	{
		this->learningRate = Scalar::make(engine);
		this->decay = Scalar::make(engine);
		// upload float to Scalar
		*learningRate = initLearningRate;
		*decay = decayRate;
	}

	virtual void setup_impl(Tensor&) {}

	// FIXME because update() is compiled, we cannot branch on different 'state'
	// 'decay' has to be per minibatch
	virtual void update_impl(
			Tensor& paramValue, Tensor& paramGradient, LearningState::Ptr state)
	{
		paramValue -= *learningRate * paramGradient;
		*learningRate *= *decay;
	}
};


/**
 * Learning with momentum
 */
class MomentumSGD : public Optimizer
{
public:
	MomentumSGD(float initLearningRate) :
		Optimizer(),
		initLearningRate(initLearningRate)
	{}

	virtual ~MomentumSGD() {}

protected:
	Scalar::Ptr learningRate;
	Tensor::Ptr lastUpdate;

	float initLearningRate;

	virtual void initialize_impl()
	{
		this->learningRate = Scalar::make(engine);
		this->lastUpdate = Tensor::make(engine);
		// upload float to Scalar
		*learningRate = initLearningRate;
	}

	virtual void setup_impl(Tensor&) {}

	virtual void update_impl(
			Tensor& paramValue, Tensor& paramGradient, LearningState::Ptr state)
	{
		paramValue -= *learningRate * paramGradient;
	}
};

#endif /* OPTIMIZER_H_ */
