/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_LISTENER_H_
#define LEARNING_LISTENER_H_

#include "utils/laminar_utils.h"

enum class LearningStage
{
	Training,
	Validation,
	Testing
};

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

	float testingLoss = 0;
	// percentage accuracy, perplexity, etc.
	float testingMetric = 0;

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
******* Evalution Schedule *********
**************************************/
/**
 * How often do we do testing/validation
 */


#endif /* LEARNING_LISTENER_H_ */
