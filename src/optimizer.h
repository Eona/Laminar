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
 * http://cs231n.github.io/neural-networks-3/#sgd
 */
class MomentumGD : public Optimizer
{
public:
	MomentumGD(float learningRate, float moment) :
		Optimizer(),
		learningRate(learningRate),
		moment(moment)
	{}

	virtual ~MomentumGD() {}

private:
	// map a param's addr to its previous update tensor (history)
	std::unordered_map<int, Tensor::Ptr> momentumMap;

protected:
	float learningRate;
	float moment;

	virtual void initialize_impl()
	{ }

	virtual void setup_impl(Tensor& param)
	{
		// initialize momentum tensors to be of the same dimension with all zeros
		momentumMap[param.addr] = Tensor::make(engine, param.dim());
	}

	virtual void update_impl(
			Tensor& paramValue, Tensor& paramGradient, LearningState::Ptr state)
	{
		Tensor& momentum = *momentumMap[paramValue.addr];

		momentum = moment * momentum - learningRate * paramGradient;
		paramValue += momentum;
	}

};

/**
 * Nesterov momentum
 * http://cs231n.github.io/neural-networks-3/#sgd
 */
class NesterovMomentum : public MomentumGD
{
public:
	NesterovMomentum(float learningRate, float moment) :
		MomentumGD(learningRate, moment)
	{}

	virtual ~NesterovMomentum() {}

private:
	// map a param's addr to its previous update tensor (history)
	std::unordered_map<int, std::array<Tensor::Ptr, 2>> momentumMap;
	// index
	enum {
		PREV = 0,
		CURRENT = 1
	};

protected:

	virtual void setup_impl(Tensor& param)
	{
		// initialize momentum tensors to be of the same dimension with all zeros
		momentumMap[param.addr][PREV] = Tensor::make(engine, param.dim());
		momentumMap[param.addr][CURRENT] = Tensor::make(engine, param.dim());
	}

	virtual void update_impl(
			Tensor& paramValue, Tensor& paramGradient, LearningState::Ptr state)
	{
		Tensor& momentumPrev = *momentumMap[paramValue.addr][PREV];
		Tensor& momentumCur = *momentumMap[paramValue.addr][CURRENT];

		/*
		v_prev = v # back this up
		v = mu * v - learning_rate * dx # velocity update stays the same
		x += -mu * v_prev + (1 + mu) * v # position update changes form
		*/
		momentumPrev = momentumCur;
		momentumCur = moment * momentumCur - learningRate * paramGradient;
		paramValue += -moment * momentumPrev + (1 + moment) * momentumCur;
	}
};

#endif /* OPTIMIZER_H_ */
