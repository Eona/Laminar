/*
 * Eona Studio (c) 2015
 */

#ifndef SERIALIZER_H_
#define SERIALIZER_H_

#include "learning_listener.h"
#include "network.h"
/**************************************
******* Serializer *********
**************************************/
/**
 * Save parameters to disk periodically
 */
template<typename NetworkT, typename EngineT>
struct Serializer
{
LMN_STATIC_ASSERT_IS_BASE(Network, NetworkT, "Serializer template arg");
LMN_STATIC_ASSERT_IS_BASE(EngineBase, EngineT, "Serializer template arg");

	virtual ~Serializer() {}

	virtual void save(std::shared_ptr<NetworkT> net, LearningState::Ptr state)
	{
		// ugly workaround because NetworkT is not known to derive from Network
		Network::Ptr net_ = Network::cast<Network>(net);
		auto engine = net_->get_engine<EngineT>();
		LMN_ASSERT_NULLPTR(engine,
			LearningException("Serializer engine type mismatch"));

		this->save_impl(net, engine, state);
	}

protected:
	// should implement this
	virtual void save_impl(
		std::shared_ptr<NetworkT>, std::shared_ptr<EngineT>, LearningState::Ptr) = 0;
};

/**
 * Doesn't save anything
 */
struct NullSerializer : public Serializer<Network, EngineBase>
{
	virtual ~NullSerializer() {}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(NullSerializer)

protected:
	virtual void save_impl(
		std::shared_ptr<Network>, std::shared_ptr<EngineBase>, LearningState::Ptr)
	{ }
};


#endif /* SERIALIZER_H_ */
