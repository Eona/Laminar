/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_SESSION_H_
#define LEARNING_SESSION_H_

#include "network.h"
#include "rnn.h"
#include "optimizer.h"
#include "learning_listener.h"
#include "evaluator.h"

template<typename NetworkT>
class LearningSessionBase
{
LMN_STATIC_ASSERT_IS_BASE(Network, NetworkT, "LearningSession template arg");

typedef std::shared_ptr<NetworkT> NetworkTPtr;

// otherwise we have to add 'typename' every time:
typedef typename Serializer<NetworkT>::Ptr SerializerPtr;

public:
	LearningSessionBase(Network::Ptr net,
						Optimizer::Ptr optimizer,
						EvaluatorBase<>::Ptr evaluator,
						StopCriteria::Ptr stopper,
						SerializerPtr serializer) :
		net(Network::cast<NetworkT>(net)),
		dataManager(net->get_data_manager()),
		engine(net->get_engine()),
		state(LearningState::make()),
		optimizer(optimizer),
		evaluator(evaluator),
		stopper(stopper),
		serializer(serializer),
		initGuard("LearningSession")
	{
		LMN_ASSERT_NULLPTR(this->net,
			LearningException("LearningSession network type mismatch"));
	}

	virtual ~LearningSessionBase() {}

	virtual void initialize()
	{
		this->initialize_impl();

		initGuard.initialize();
	}

	void init_total_epoch(int totalEpoch)
	{
		initGuard.assert_before_initialize("init_total_epoch");
		state->totalEpoch = totalEpoch;
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

		// Monitor dataManager current_epoch change
		ChangeMonitor<int> currentEpochMon([this]() {
			return dataManager->current_epoch();
		});

		do {
			evaluator->set_learning_stage(LearningStage::Training);

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

			++ state->currentBatch;
			state->currentEpoch = dataManager->current_epoch();

			// If we finish another epoch
			if (currentEpochMon.monitor())
			{
				state->currentBatch = 0; // new epoch reset batch count
				state->trainingLoss = evaluator->net_loss();

				// We do optional validation/testing at the end of each epoch
				/*********** Validation ***********/
				evaluator->set_learning_stage(LearningStage::Validation);
				evaluator->validation();
				state->validationLoss = evaluator->validation_loss();
				state->validationMetric = evaluator->validation_metric();

				/*********** Testing ***********/
				evaluator->set_learning_stage(LearningStage::Testing);
				evaluator->testing();
				state->testingLoss = evaluator->testing_loss();
				state->testingMetric = evaluator->testing_metric();

				// Save parameters to disk
				serializer->save(net, state);
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
	NetworkTPtr net;
	DataManagerBase::Ptr dataManager;
	EngineBase::Ptr engine;

	LearningState::Ptr state;
	Optimizer::Ptr optimizer;
	EvaluatorBase<>::Ptr evaluator;
	StopCriteria::Ptr stopper;
	SerializerPtr serializer;

	InitializeGuard<LearningException> initGuard;

	vector<ParamContainer::Ptr> paramContainers;
};

template<typename NetworkT>
class LearningSession :
		public LearningSessionBase<NetworkT>
{
typedef typename Serializer<NetworkT>::Ptr SerializerPtr;
public:
	LearningSession(Network::Ptr net,
					Optimizer::Ptr optimizer,
					EvaluatorBase<>::Ptr evaluator,
					StopCriteria::Ptr stopper,
					SerializerPtr serializer) :
		LearningSessionBase<NetworkT>(
				net, optimizer, evaluator, stopper, serializer)
	{ }
};

/**
 * Specialization for recurrent net
 */
template<>
class LearningSession<RecurrentNetwork> :
		public LearningSessionBase<RecurrentNetwork>
{
typedef typename Serializer<RecurrentNetwork>::Ptr SerializerPtr;

public:
	LearningSession(Network::Ptr net,
					Optimizer::Ptr optimizer,
					EvaluatorBase<>::Ptr evaluator,
					StopCriteria::Ptr stopper,
					SerializerPtr serializer) :
		LearningSessionBase<RecurrentNetwork>(
				net, optimizer, evaluator, stopper, serializer)
	{ }

};

#endif /* LEARNING_SESSION_H_ */
