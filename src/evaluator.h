/*
 * Eona Studio (c) 2015
 */

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "network.h"

/**************************************
******* Cross-validation and testing *********
**************************************/
// For static template arg type checking only
struct EvaluatorBase
{
	virtual ~EvaluatorBase() {}
};

/**
 * Evaluate network performance for validation and testing
 */
template<typename EngineT, typename FloatT = float>
class Evaluator : public EvaluatorBase
{
LMN_STATIC_ASSERT_IS_BASE(EngineBase, EngineT, "Evaluator template arg");

public:
	Evaluator(Network::Ptr net) :
		net(net),
		engine(net->get_engine<EngineT>()),
		dataManager(net->get_data_manager()),
		losses({ 0, 0, 0 }), // validation/testing losses
		metrics({ 0, 0, 0 }) // validation/testing metrics
	{}

	virtual ~Evaluator() {}

	FloatT read_network_loss()
	{
		return engine->scalar_at(net->loss_value());
	}

	virtual void evaluate(LearningPhase learnPhase)
	{
		LMN_ASSERT_THROW(learnPhase != LearningPhase::Training,
			LearningException("Evaluator cannot run 'Training' phase"));

		int phase = enum2integral(learnPhase);

		this->losses[phase] = 0;
		this->metrics[phase] = 0;

		dataManager->set_learning_phase(learnPhase);

		int totalBatches = 0; // #batches processed
		float totalEvalLoss = 0; // running total

		while (!dataManager->is_end_of_epoch())
		{
			net->execute("load_input");
			net->execute("load_target");

			// Only does forward prop
			net->execute("forward");

			/*********** Update validation results ***********/
			totalBatches += dataManager->batch_size();
			totalEvalLoss += this->read_network_loss();

			this->update_metric(net, learnPhase);

			/*********** Prepare for next validation epoch ***********/
			net->execute("zero_clear");
			dataManager->prepare_next_batch();
		}

		this->losses[phase] = totalEvalLoss / totalBatches;
		this->metrics[phase] = this->summarize_metric(net, learnPhase);

		// Reset the validation stream
		dataManager->reset_epoch();
	}

	FloatT loss(LearningPhase learnPhase) const
	{
		return this->losses[enum2integral(learnPhase)];
	}

	FloatT metric(LearningPhase learnPhase) const
	{
		return this->metrics[enum2integral(learnPhase)];
	}

protected:
	/**
	 * Derived needs to implement this
	 * will be called after every minibatch to update whatever metric
	 * NOTE a metric is not the same as loss value, e.g. percentage accuracy.
	 */
	virtual void update_metric(Network::Ptr, LearningPhase) = 0;

	/**
	 * will be called at the end of validation/testing to get a summary metric
	 */
	virtual FloatT summarize_metric(Network::Ptr, LearningPhase) = 0;

protected:
	Network::Ptr net;
	std::shared_ptr<EngineT> engine;
	DataManagerBase::Ptr dataManager;

private:
	std::array<FloatT, LEARNING_PHASE_N> losses;
	std::array<FloatT, LEARNING_PHASE_N> metrics;
};


template<typename EngineT, typename FloatT = float>
struct NoMetricEvaluator : public Evaluator<EngineT, FloatT>
{
	NoMetricEvaluator(Network::Ptr net) :
		Evaluator<EngineT, FloatT>(net)
	{ }

	virtual ~NoMetricEvaluator() {}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(NoMetricEvaluator)

protected:

	/*********** defaults ***********/
	virtual void update_metric(Network::Ptr, LearningPhase)
	{}

	virtual FloatT summarize_metric(Network::Ptr, LearningPhase)
	{
		return 0;
	}
};

#endif /* EVALUATOR_H_ */
