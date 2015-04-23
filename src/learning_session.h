/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_SESSION_H_
#define LEARNING_SESSION_H_

#include "network.h"
#include "rnn.h"
#include "optimizer.h"
#include "learning_listener.h"

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
						StopCriteria::Ptr stopper,
						SerializerPtr serializer) :
		net(Network::cast<NetworkT>(net)),
		dataManager(net->get_data_manager()),
		engine(net->get_engine()),
		state(LearningState::make()),
		optimizer(optimizer),
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

//		for (int epoch = 0; epoch < totalEpoch; ++epoch)
		{
			net->execute("load_input");
			net->execute("load_target");

			net->execute("forward");
			net->execute("backward");

			for (auto pc : paramContainers)
				optimizer->update(pc, state);
			engine->flush_execute();
		}

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
					StopCriteria::Ptr stopper,
					SerializerPtr serializer) :
		LearningSessionBase<NetworkT>(
				net, optimizer, stopper, serializer)
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
					StopCriteria::Ptr stopper,
					SerializerPtr serializer) :
		LearningSessionBase<RecurrentNetwork>(
				net, optimizer, stopper, serializer)
	{ }

};

#endif /* LEARNING_SESSION_H_ */
