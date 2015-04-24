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
struct SerializerBase
{
	virtual ~SerializerBase() {};

	virtual void save(Network::Ptr, LearningState::Ptr) = 0;

	TYPEDEF_PTR(SerializerBase);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(SerializerBase)
};

template<typename NetworkT, typename EngineT>
struct Serializer : public SerializerBase
{
LMN_STATIC_ASSERT_IS_BASE(Network, NetworkT, "Serializer template arg");
LMN_STATIC_ASSERT_IS_BASE(EngineBase, EngineT, "Serializer template arg");

	virtual ~Serializer() {}

	virtual void save(Network::Ptr net_, LearningState::Ptr state)
	{
		auto net = Network::cast<NetworkT>(net_);
		LMN_ASSERT_NULLPTR(net,
			LearningException("Serializer network type mismatch"));

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
struct NullSerializer : public SerializerBase
{
	void save(Network::Ptr, LearningState::Ptr) { }

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(NullSerializer)
};


#endif /* SERIALIZER_H_ */
