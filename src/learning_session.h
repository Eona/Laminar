/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_SESSION_H_
#define LEARNING_SESSION_H_

#include "network.h"
#include "rnn.h"
#include "optimizer.h"
#include "learning_listener.h"
#include "serializer.h"
#include "evaluator.h"

template<typename OptimizerT,
		typename EvaluatorT,
		typename EvalScheduleT,
		typename StopCriteriaT,
		typename SerializerT,
		typename ObserverT>
class LearningSession
{
	template<typename T>
	using PtrT = std::shared_ptr<T>;

public:
	LearningSession(Network::Ptr net,
					PtrT<OptimizerT> optimizer,
					PtrT<EvaluatorT> evaluator,
					PtrT<EvalScheduleT> schedule,
					PtrT<StopCriteriaT> stopper,
					PtrT<SerializerT> serializer,
					PtrT<ObserverT> observer) :
		net(net),
		dataManager(net->get_data_manager()),
		engine(net->get_engine()),
		state(LearningState::make()),
		optimizer(optimizer),
		evaluator(evaluator),
		schedule(schedule),
		stopper(stopper),
		serializer(serializer),
		observer(observer),
		initGuard("LearningSession")
	{ }

	virtual ~LearningSession() {}

	virtual void initialize()
	{
		this->initialize_impl();

		initGuard.initialize();
	}

	/**
	 * Main training entry function
	 */
	virtual void train()
	{
		initGuard.assert_after_initialize("train");

		// Monitor dataManager batch size change
		ChangeMonitor<int> batchSizeMon([this]() {
			return dataManager->batch_size();
		});

		float totalTrainingLoss = 0; // keep a running total

		do {
			dataManager->set_learning_phase(LearningPhase::Training);

			LMN_ASSERT_THROW(!batchSizeMon.monitor(),
				UnimplementedException(
					"LearningSession doesn't support changing batch size for now.\n"
					"batch_size has changed from " + to_str(batchSizeMon.previous()) +
					" to " + to_str(batchSizeMon.current())));

			state->batchSize = dataManager->batch_size();

			net->execute("load_input");
			net->execute("load_target");

			net->execute("forward");
			net->execute("backward");

			/********* optionally observe network at every minibatch ********/
			observer->observe(net, state);

			/*********** Update state after this minibatch ***********/
			// all batches processed so far
			state->batchAll += state->batchSize;
			// current batch in the current epoch
			state->batchInEpoch += state->batchSize;
			// Accumulate batch loss to this epoch's total training loss
			totalTrainingLoss += evaluator->read_network_loss();
			// running average
			state->trainingLoss = totalTrainingLoss / state->batchInEpoch;

			/*********** Update parameters ***********/
			for (auto pc : net->param_containers())
				optimizer->update(pc, state);
			engine->flush_execute();

			/*********** Prepare for next minibatch ***********/
			// zero clears all in/out values/gradients and loss value
			net->execute("zero_clear");
			dataManager->prepare_next_batch();

			// If we finish another epoch
			if (dataManager->is_end_of_epoch())
			{
				// We do optional validation/testing at the end of each epoch
				/*********** Validation ***********/
				if (schedule->run_validation(state))
				{
					auto phase = LearningPhase::Validation;
					evaluator->evaluate(phase);
					state->validationLoss = evaluator->loss(phase);
					state->validationMetric = evaluator->metric(phase);
				}

				/*********** Testing ***********/
				if (schedule->run_testing(state))
				{
					auto phase = LearningPhase::Testing;
					evaluator->evaluate(phase);
					state->testingLoss = evaluator->loss(phase);
					state->testingMetric = evaluator->metric(phase);
				}

				DEBUG_MSG("Epoch", state->epoch);
				DEBUG_MSG("Training loss", state->trainingLoss);
				DEBUG_MSG("Validation loss", state->validationLoss);
				DEBUG_MSG("Testing loss", state->testingLoss);

				/*********** Save to disk ***********/
				serializer->save(net, state);

				/*********** Record learning history ***********/
				// copy and store to history vector
				learningHistory.push_back(*state);

				/*********** Prepare for the next epoch ***********/
				dataManager->set_learning_phase(LearningPhase::Training);
				dataManager->reset_epoch();
				++ state->epoch;
				// zero clear some stats
				state->batchInEpoch = 0; // new epoch reset batch count
				state->clear_loss();
				totalTrainingLoss = 0.f;
				net->execute("zero_clear");
			}
		}
		while (!stopper->stop_learning(state));
	}

	TYPEDEF_PTR(LearningSession);

protected:
	/**
	 * Subclasses should override this
	 */
	virtual void initialize_impl()
	{
		net->execute("initialize");

		optimizer->init_engine(engine);
		optimizer->initialize();
		// FIXME unify 'execute' interface of Network & Optimizer
		engine->flush_execute();
	}

protected:
	Network::Ptr net;
	DataManagerBase::Ptr dataManager;
	EngineBase::Ptr engine;

	LearningState::Ptr state;
	PtrT<OptimizerT> optimizer;
	PtrT<EvaluatorT> evaluator;
	PtrT<EvalScheduleT> schedule;
	PtrT<StopCriteriaT> stopper;
	PtrT<SerializerT> serializer;
	PtrT<ObserverT> observer;

	InitializeGuard<LearningException> initGuard;

	vector<LearningState> learningHistory;
};

/**
 * In analogy to std::make_shared, we use a factory method
 * to deduce the class templates
 */
#define LEARNING_SESSION_GENERIC_TYPE \
LearningSession<OptimizerT, EvaluatorT, EvalScheduleT, StopCriteriaT, SerializerT, ObserverT>

template<typename OptimizerT,
		typename EvaluatorT,
		typename EvalScheduleT,
		typename StopCriteriaT,
		typename SerializerT,
		typename ObserverT>
typename LEARNING_SESSION_GENERIC_TYPE::Ptr
new_learning_session(Network::Ptr net,
		std::shared_ptr<OptimizerT> optimizer, // actually should be pointer types
		std::shared_ptr<EvaluatorT> evaluator,
		std::shared_ptr<EvalScheduleT> schedule,
		std::shared_ptr<StopCriteriaT> stopper,
		std::shared_ptr<SerializerT> serializer,
		std::shared_ptr<ObserverT> observer)
{
	return std::make_shared<LEARNING_SESSION_GENERIC_TYPE>(
				net,
				optimizer,
				evaluator,
				schedule,
				stopper,
				serializer,
				observer);
}


#endif /* LEARNING_SESSION_H_ */
