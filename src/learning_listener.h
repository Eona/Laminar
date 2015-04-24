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
struct EvalSchedule
{
	virtual ~EvalSchedule() {}

	/**
	 * Called at the end of every epoch (epoch not incremented yet)
	 * @return true if you want to run validation
	 */
	virtual bool run_validation(LearningState::Ptr) = 0;

	/**
	 * Called at the end of every epoch (epoch not incremented yet)
	 * @return true if you want to run testing
	 */
	virtual bool run_testing(LearningState::Ptr) = 0;

	TYPEDEF_PTR(EvalSchedule);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(EvalSchedule)
};

// Support for 'diamond' inheritance: virtual inheritance
// e.g. ValidationOnceNoTestingSchedule : public ValidationSchedule, NoTestingSchedule
/**
 * Doesn't run testing
 */
struct NoTestingSchedule : public virtual EvalSchedule
{
	virtual ~NoTestingSchedule() {}

	virtual bool run_testing(LearningState::Ptr)
	{
		return false;
	}
};

/**
 * Doesn't run validation
 */
struct NoValidationSchedule : public virtual EvalSchedule
{
	virtual ~NoValidationSchedule() {}

	virtual bool run_validation(LearningState::Ptr)
	{
		return false;
	}
};

/**
 * Validate/test once every n epoch
 */
struct EpochIntervalSchedule : public EvalSchedule
{
	/**
	 * @param validationInterval if 0, never validate
	 * @param testInterval if 0, never test
	 */
	EpochIntervalSchedule(int validationInterval = 0, int testInterval = 0) :
			validationInterval(validationInterval),
			testInterval(testInterval)
	{}

	virtual ~EpochIntervalSchedule() {}

	virtual bool run_validation(LearningState::Ptr state)
	{
		if (validationInterval <= 0)
			return false;
		return (state->currentEpoch + 1) % validationInterval == 0;
	}

	virtual bool run_testing(LearningState::Ptr state)
	{
		if (testInterval <= 0)
			return false;
		return (state->currentEpoch + 1) % testInterval == 0;
	}

protected:
	int validationInterval;
	int testInterval;
};

#endif /* LEARNING_LISTENER_H_ */
