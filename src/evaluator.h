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
		net(net),
		dataManager(net->get_data_manager())
	{}

	virtual ~EvaluatorBase() {}

	virtual FloatT network_loss() = 0;

	void validation()
	{
//		LMN_ASSERT_THROW(learning_phase() == LearningPhase::Validation,
//			LearningException("LearningPhase must be 'Validation' "
//				"for Evaluator::validation(). Use set_learning_phase() to switch."));
		dataManager->set_learning_phase(LearningPhase::Validation);

		this->validationMetric = this->validation_impl(net);
		this->validationLoss = this->network_loss();

		// Reset the validation stream
		// FIXME
		dataManager->reset_epoch();
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
//		LMN_ASSERT_THROW(learning_phase() == LearningPhase::Testing,
//			LearningException("LearningPhase must be 'Testing' "
//				"for Evaluator::testing(). Use set_learning_phase() to switch."));
		dataManager->set_learning_phase(LearningPhase::Testing);

		this->testingMetric = this->testing_impl(net);
		this->testingLoss = this->network_loss();

		// Reset the testing stream
		// FIXME
		dataManager->reset_epoch();
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
	/**
	 * Return a validation metric
	 * NOTE not necessary the same as validation loss.
	 * E.g. a metric can be percentage accuracy.
	 */
	virtual FloatT validation_impl(Network::Ptr) = 0;

	/**
	 * Return a testing metric
	 * NOTE not necessary the same as testing loss.
	 * E.g. a metric can be percentage accuracy.
	 */
	virtual FloatT testing_impl(Network::Ptr) = 0;

protected:
	Network::Ptr net;
	DataManagerBase::Ptr dataManager;

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

	FloatT network_loss()
	{
		return engine->scalor_at(net->loss_value());
	}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(Evaluator)

protected:
	std::shared_ptr<EngineT> engine;

	/**
	 * Overridable default implementation
	 */
	virtual FloatT validation_impl(Network::Ptr net)
	{
		return 0; // FIXME
	}

	virtual FloatT testing_impl(Network::Ptr net)
	{
		return 0;
	}
};



#endif /* EVALUATOR_H_ */
