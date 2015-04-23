/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_LISTENER_H_
#define LEARNING_LISTENER_H_

#include "utils/laminar_utils.h"

struct LearningState
{
	virtual ~LearningState() {}

	int currentEpoch = 0;
	int totalEpoch;

	int currentBatch = 0;
	int batchSize;

	// from network's loss function
	float trainingLoss = 0;
	float validationLoss = 0;

	// percentage accuracy, perplexity, etc.
	float validationMetric = 0;

	TYPEDEF_PTR(LearningState);

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(LearningState)
};


/**************************************
******* StopCriteria *********
**************************************/
/**
 * When to stop learning
 */
struct StopCriteria
{
	virtual ~StopCriteria() {}

	/**
	 * @return true to stop learning
	 */
	virtual bool stop_learning(LearningState::Ptr) = 0;

	TYPEDEF_PTR(StopCriteria);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(StopCriteria)
};

struct EpochStopCriteria : public StopCriteria
{
	virtual bool stop_learning(LearningState::Ptr state)
	{
		return state->currentEpoch >= state->totalEpoch;
	}
};

/**************************************
******* Serializer *********
**************************************/
// forward decl
class Network;
/**
 * Save parameters to disk periodically
 */
template<typename NetworkT>
struct Serializer
{
LMN_STATIC_ASSERT_IS_BASE(Network, NetworkT, "Serializer template arg");

	virtual ~Serializer() {}

	virtual void save(std::shared_ptr<NetworkT>, LearningState::Ptr) = 0;

	TYPEDEF_PTR(Serializer<NetworkT>);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(Serializer<NetworkT>)
};

/**
 * Doesn't save anything
 */
template<typename NetworkT>
struct NullSerializer : public Serializer<NetworkT>
{
	void save(std::shared_ptr<NetworkT>, LearningState::Ptr) { }

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(NullSerializer<NetworkT>)
};

#endif /* LEARNING_LISTENER_H_ */
