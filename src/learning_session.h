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
		typename StopCriteriaT = MaxEpochStopper,
		typename SerializerT = NullSerializer,
		typename EvalScheduleT = NullSchedule,
		typename ObserverT = NullObserver>
class LearningSession
{
LMN_STATIC_ASSERT_IS_BASE(Optimizer, OptimizerT, "LearningSession template arg #1");
LMN_STATIC_ASSERT_IS_BASE(EvaluatorBase, EvaluatorT, "LearningSession template arg #2");
LMN_STATIC_ASSERT_IS_BASE(StopCriteria, StopCriteriaT, "LearningSession template arg #3");
LMN_STATIC_ASSERT_IS_BASE(SerializerBase, SerializerT, "LearningSession template arg #4");
LMN_STATIC_ASSERT_IS_BASE(EvalSchedule, EvalScheduleT, "LearningSession template arg #5");
LMN_STATIC_ASSERT_IS_BASE(ObserverBase, ObserverT, "LearningSession template arg #6");

public:
	LearningSession(Network::Ptr net,
				std::shared_ptr<OptimizerT> optimizer,
				std::shared_ptr<EvaluatorT> evaluator,
				std::shared_ptr<StopCriteriaT> stopper,
				std::shared_ptr<SerializerT> serializer,
				std::shared_ptr<EvalScheduleT> schedule,
				std::shared_ptr<ObserverT> observer) :
		net(net),
		dataManager(net->get_data_manager()),
		engine(net->get_engine()),
		state(LearningState::make()),
		optimizer(optimizer),
		evaluator(evaluator),
		stopper(stopper),
		serializer(serializer),
		schedule(schedule),
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
		// FIXME unify execution map interface
		Routine::Ptr optimizerRoutine;

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

			/*********** Update state after this minibatch ***********/
			// all batches processed so far
			state->batchAll += state->batchSize;
			// current batch in the current epoch
			state->batchInEpoch += state->batchSize;
			// Accumulate batch loss to this epoch's total training loss
			float curBatchLoss = evaluator->read_network_loss();
			totalTrainingLoss += curBatchLoss;
			// running average
			state->trainingLoss = totalTrainingLoss / state->batchInEpoch;
			state->curBatchTrainingLoss = curBatchLoss / state->batchSize;

			/********* optionally observe network at every minibatch ********/
			observer->observe(net, state);

			/*********** Update parameters ***********/
			if (!optimizerRoutine)
			{
				optimizer->update(net->param_containers(), state);
				optimizerRoutine = engine->flush_execute();
			}
			else
				optimizerRoutine->execute();

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
				DEBUG_MSG("Validation metric", state->validationMetric);
				DEBUG_MSG("Testing loss", state->testingLoss);
				DEBUG_MSG("Testing metric", state->testingMetric);

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
		optimizer->initialize(net->param_containers());

		// FIXME unify 'execute' interface of Network & Optimizer
		engine->flush_execute();
	}

protected:
	Network::Ptr net;
	DataManagerBase::Ptr dataManager;
	EngineBase::Ptr engine;

	LearningState::Ptr state;
	std::shared_ptr<OptimizerT> optimizer;
	std::shared_ptr<EvaluatorT> evaluator;
	std::shared_ptr<StopCriteriaT> stopper;
	std::shared_ptr<SerializerT> serializer;
	std::shared_ptr<EvalScheduleT> schedule;
	std::shared_ptr<ObserverT> observer;

	InitializeGuard<LearningException> initGuard;

	vector<LearningState> learningHistory;
};

/**
 * In analogy to std::make_shared, we use a factory method
 * to deduce the class templates
 */
#define LEARNING_SESSION_GENERIC_TYPE \
LearningSession<OptimizerT, EvaluatorT, StopCriteriaT, SerializerT, EvalScheduleT, ObserverT>

/**
 *
 * @param net
 * @param optimizer
 * @param evaluator
 * @param stopper
 * @param serializer if omitted, will be NullSerializer
 * @param schedule if omitted, will be NullSchedule
 * @param observer if omitted, will be NullObserver
 * @return
 */
template<typename OptimizerT,
		typename EvaluatorT,
		typename StopCriteriaT,
		typename SerializerT = NullSerializer,
		typename EvalScheduleT = NullSchedule,
		typename ObserverT = NullObserver>
typename LEARNING_SESSION_GENERIC_TYPE::Ptr
new_learning_session(Network::Ptr net,
		std::shared_ptr<OptimizerT> optimizer, // actually should be pointer types
		std::shared_ptr<EvaluatorT> evaluator,
		std::shared_ptr<StopCriteriaT> stopper,
		std::shared_ptr<SerializerT> serializer = nullptr,
		std::shared_ptr<EvalScheduleT> schedule = nullptr,
		std::shared_ptr<ObserverT> observer = nullptr)
{
	// default options
	if (!serializer)
	{
		// NOTE workaround to get default type to work
		// serializer must have type SerializerT (can be anything),
		// so we have to force a cast from NullSerializer
		serializer = std::dynamic_pointer_cast<SerializerT>(
							std::make_shared<NullSerializer>());
		LMN_ASSERT_NULLPTR(serializer,
			LearningException("Invalid Serializer default arg"));
	}

	if (!schedule)
	{
		schedule = std::dynamic_pointer_cast<EvalScheduleT>(
							std::make_shared<NullSchedule>());
		LMN_ASSERT_NULLPTR(schedule,
			LearningException("Invalid EvalSchedule default arg"));
	}

	if (!observer)
	{
		observer = std::dynamic_pointer_cast<ObserverT>(
							std::make_shared<NullObserver>());
		LMN_ASSERT_NULLPTR(observer,
			LearningException("Invalid Observer default arg"));
	}

	return std::make_shared<LEARNING_SESSION_GENERIC_TYPE>(
					net,
					optimizer,
					evaluator,
					stopper,
					serializer,
					schedule,
					observer);
}


#endif /* LEARNING_SESSION_H_ */
