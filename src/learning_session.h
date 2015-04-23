/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_SESSION_H_
#define LEARNING_SESSION_H_

#include "network.h"
#include "rnn.h"
#include "optimizer.h"

template<typename NetworkT>
class LearningSessionBase
{
LMN_STATIC_ASSERT((std::is_base_of<Network, NetworkT>::value),
		"LearningSession type paramater must be a subclass of Network");

typedef std::shared_ptr<NetworkT> NetworkTPtr;

public:
	LearningSessionBase(Network::Ptr net, Optimizer::Ptr optimizer) :
		net(Network::cast<NetworkT>(net)),
		dataManager(net->get_data_manager()),
		engine(net->get_engine()),
		optimizer(optimizer),
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

	virtual void train(int totalEpoch)
	{
		initGuard.assert_after_initialize("train");

//		for (int epoch = 0; epoch < totalEpoch; ++epoch)
		{
			net->execute("load_input");
			net->execute("load_target");

			net->execute("forward");
			net->execute("backward");

			for (auto pc : paramContainers)
				optimizer->update(pc);
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
	Optimizer::Ptr optimizer;

	InitializeGuard<LearningException> initGuard;

	int totalEpoch;
	vector<ParamContainer::Ptr> paramContainers;
};

template<typename NetworkT>
class LearningSession :
		public LearningSessionBase<NetworkT>
{
public:
	LearningSession(Network::Ptr net, Optimizer::Ptr optimizer) :
		LearningSessionBase<NetworkT>(net, optimizer)
	{ }
};

/**
 * Specialization for recurrent net
 */
template<>
class LearningSession<RecurrentNetwork> :
		public LearningSessionBase<RecurrentNetwork>
{
public:
	LearningSession(Network::Ptr net, Optimizer::Ptr optimizer) :
		LearningSessionBase<RecurrentNetwork>(net, optimizer)
	{ }

};

#endif /* LEARNING_SESSION_H_ */
