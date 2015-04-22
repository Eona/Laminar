/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_SESSION_H_
#define LEARNING_SESSION_H_

#include "network.h"
#include "rnn.h"

template<typename NetworkT>
class LearningSessionBase
{
LMN_STATIC_ASSERT((std::is_base_of<Network, NetworkT>::value),
		"LearningSession type paramater must be a subclass of Network");

typedef std::shared_ptr<NetworkT> NetworkTPtr;

public:
	LearningSessionBase(Network::Ptr network_):
		network(Network::cast<NetworkT>(network_)),
		dataManager(network_->get_data_manager())
	{
		LMN_ASSERT_NULLPTR(network,
			LearningException("network type mismatch"));
	}

	virtual ~LearningSessionBase() {}



protected:
	NetworkTPtr network;
	DataManagerBase::Ptr dataManager;
};

template<typename NetworkT>
class LearningSession :
		public LearningSessionBase<NetworkT>
{
public:
	LearningSession(Network::Ptr network) :
		LearningSessionBase<NetworkT>(network)
	{ }
};

/**
 * Specialization for recurrent network
 */
template<>
class LearningSession<RecurrentNetwork> :
		public LearningSessionBase<RecurrentNetwork>
{
public:
	LearningSession(Network::Ptr network) :
		LearningSessionBase<RecurrentNetwork>(network)
	{ }

};

#endif /* LEARNING_SESSION_H_ */
