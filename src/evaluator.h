/*
 * Eona Studio (c) 2015
 */

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "network.h"

/**************************************
******* Cross-validation and testing *********
**************************************/

template<typename FloatT = float>
class EvaluatorBase
{
public:
	EvaluatorBase(Network::Ptr net) :
		net(net)
	{}

	virtual ~EvaluatorBase() {}

	virtual FloatT net_loss() = 0;

	void set_learning_stage(LearningStage dataStage)
	{
		net->get_data_manager()->set_learning_stage(dataStage);
	}

	LearningStage learning_stage()
	{
		return net->get_data_manager()->learning_stage();
	}

	void validation()
	{
		LMN_ASSERT_THROW(learning_stage() == LearningStage::Validation,
			LearningException("LearningStage must be 'Validation' "
				"for Evaluator::validation(). Use set_learning_stage() to switch."));

		this->validationMetric = this->validation_impl();
		this->validationLoss = this->net_loss();
	}

	FloatT validation_loss() const
	{
		return this->validationLoss;
	}

	FloatT validation_metric() const
	{
		return this->validationMetric;
	}

	void testing()
	{
		LMN_ASSERT_THROW(learning_stage() == LearningStage::Testing,
			LearningException("LearningStage must be 'Testing' "
				"for Evaluator::testing(). Use set_learning_stage() to switch."));

		this->testingMetric = this->testing_impl();
		this->testingLoss = this->net_loss();
	}

	FloatT testing_loss() const
	{
		return this->testingLoss;
	}

	FloatT testing_metric() const
	{
		return this->testingMetric;
	}

	TYPEDEF_PTR(EvaluatorBase);

protected:
	Network::Ptr net;

	/**
	 * Return a validation metric
	 * NOTE not necessary the same as validation loss.
	 * E.g. a metric can be percentage accuracy.
	 */
	virtual FloatT validation_impl() = 0;

	/**
	 * Return a testing metric
	 * NOTE not necessary the same as testing loss.
	 * E.g. a metric can be percentage accuracy.
	 */
	virtual FloatT testing_impl() = 0;

private:
	FloatT validationLoss;
	FloatT validationMetric;
	FloatT testingLoss;
	FloatT testingMetric;
};

/**
 * Evaluate network performance for validation and testing
 */
template<typename EngineT, typename FloatT = float>
class Evaluator : public EvaluatorBase<FloatT>
{
LMN_STATIC_ASSERT_IS_BASE(EngineBase, EngineT, "Evaluator template arg");

using EvaluatorBase<FloatT>::net;

public:
	Evaluator(Network::Ptr net) :
		EvaluatorBase<FloatT>(net),
		engine(net->get_engine<EngineT>())
	{}

	virtual ~Evaluator() {}

	FloatT net_loss()
	{
		return engine->scalor_at(net->get_total_loss());
	}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(Evaluator)

protected:
	std::shared_ptr<EngineT> engine;

	/**
	 * Overridable default implementation
	 */
	virtual FloatT validation_impl()
	{
		return 0; // FIXME
	}

	virtual FloatT testing_impl()
	{
		return 0;
	}
};



#endif /* EVALUATOR_H_ */
