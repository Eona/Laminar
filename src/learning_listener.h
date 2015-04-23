/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_LISTENER_H_
#define LEARNING_LISTENER_H_

#include "utils/laminar_utils.h"

struct LearningState
{
	int currentEpoch;
	int totalEpoch;

	int currentBatch;
	int batchSize;

	float trainingLoss;
	float validationLoss;

	TYPEDEF_PTR(LearningState);

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(LearningState)
};

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
};

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

	virtual void serialize(std::shared_ptr<NetworkT>, LearningState::Ptr) = 0;
};




#endif /* LEARNING_LISTENER_H_ */
