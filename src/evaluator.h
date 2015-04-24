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
		dataManager(net->get_data_manager()),
		losses({ 0, 0, 0 }), // validation/testing losses
		metrics({ 0, 0, 0 }) // validation/testing metrics
	{}

	virtual ~EvaluatorBase() {}

	virtual FloatT read_network_loss() = 0;

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

	TYPEDEF_PTR(EvaluatorBase);

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
	DataManagerBase::Ptr dataManager;

private:
	std::array<FloatT, LEARNING_PHASE_N> losses;
	std::array<FloatT, LEARNING_PHASE_N> metrics;
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

	FloatT read_network_loss()
	{
		return engine->scalor_at(net->loss_value());
	}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(Evaluator)

protected:
	std::shared_ptr<EngineT> engine;

	/*********** defaults ***********/
	virtual void update_metric(Network::Ptr, LearningPhase)
	{}

	virtual FloatT summarize_metric(Network::Ptr, LearningPhase)
	{
		return 0;
	}
};



#endif /* EVALUATOR_H_ */
