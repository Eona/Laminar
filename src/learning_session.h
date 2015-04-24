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

class LearningSession
{
// otherwise we have to add 'typename' every time:
//typedef typename Serializer<NetworkT>::Ptr SerializerPtr;

public:
	LearningSession(Network::Ptr net,
					Optimizer::Ptr optimizer,
					EvaluatorBase<>::Ptr evaluator,
					EvalSchedule::Ptr schedule,
					StopCriteria::Ptr stopper,
					SerializerBase::Ptr serializer) :
		net(net),
		dataManager(net->get_data_manager()),
		engine(net->get_engine()),
		state(LearningState::make()),
		optimizer(optimizer),
		evaluator(evaluator),
		schedule(schedule),
		stopper(stopper),
		serializer(serializer),
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

		do {
			DEBUG_MSG("Training loop");

			dataManager->set_learning_stage(LearningStage::Training);

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

			for (auto pc : paramContainers)
				optimizer->update(pc, state);
			engine->flush_execute();

			// all batches processed so far
			++ state->batchAll;

			// current batch in the current epoch
			++ state->batchInEpoch;

			// If we finish another epoch
			if (dataManager->is_epoch_end())
			{
				state->trainingLoss = evaluator->network_loss();

				// We do optional validation/testing at the end of each epoch
				/*********** Validation ***********/
				if (schedule->run_validation(state))
				{
					evaluator->validation();
					state->validationLoss = evaluator->validation_loss();
					state->validationMetric = evaluator->validation_metric();
				}

				/*********** Testing ***********/
				if (schedule->run_testing(state))
				{
					evaluator->testing();
					state->testingLoss = evaluator->testing_loss();
					state->testingMetric = evaluator->testing_metric();
				}

				/*********** Save to disk ***********/
				serializer->save(net, state);

				/*********** Record learning history ***********/
				// copy and store to history vector
				learningHistory.push_back(*state);

				/*********** Prepare for the next epoch ***********/
				dataManager->reset_epoch(LearningStage::Training);
				++ state->epoch;
				state->batchInEpoch = 0; // new epoch reset batch count
				state->clear_loss();
			}
		}
		while (!stopper->stop_learning(state));
	}

protected:
	/**
	 * Subclasses should override this
	 */
	virtual void initialize_impl()
	{
		net->execute("initialize");

		this->paramContainers = net->get_param_containers();

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
	Optimizer::Ptr optimizer;
	EvaluatorBase<>::Ptr evaluator;
	EvalSchedule::Ptr schedule;
	StopCriteria::Ptr stopper;
	SerializerBase::Ptr serializer;

	InitializeGuard<LearningException> initGuard;

	vector<ParamContainer::Ptr> paramContainers;

	vector<LearningState> learningHistory;
};

#endif /* LEARNING_SESSION_H_ */
