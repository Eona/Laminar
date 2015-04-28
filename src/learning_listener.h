/*
 * Eona Studio (c) 2015
 */

#ifndef LEARNING_LISTENER_H_
#define LEARNING_LISTENER_H_

#include "utils/laminar_utils.h"

enum class LearningPhase
{
	Training,
	Validation,
	Testing
};

// 3 learning phases
static constexpr const int LEARNING_PHASE_N = 3;

struct LearningState
{
	virtual ~LearningState() {}

	int epoch = 0;

	int batchInEpoch = 0;
	int batchAll = 0;

	int batchSize;

	/**
	 * Training loss averaged over number of batches in the current epoch
	 */
	float trainingLoss = 0;
	/**
	 * Current minibatch training loss
	 */
	float curBatchTrainingLoss = 0;

	/**
	 * Validation loss averaged over number of batches in the current epoch
	 */
	float validationLoss = 0;
	// percentage accuracy, perplexity, etc.
	float validationMetric = 0;

	/**
	 * Testing loss averaged over number of batches in the current epoch
	 */
	float testingLoss = 0;
	// percentage accuracy, perplexity, etc.
	float testingMetric = 0;

	void clear_loss()
	{
		trainingLoss = 0;
		curBatchTrainingLoss = 0;
		validationLoss = 0;
		validationMetric = 0;
		testingLoss = 0;
		testingMetric = 0;
	}

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

/**
 * Stop when max number of epochs reached
 */
struct MaxEpochStopper : public StopCriteria
{
	/**
	 * Optionally stop when max number of processed batches reached
	 * Set to 0 to ignore maxBatch
	 */
	MaxEpochStopper(int maxEpoch, int maxBatch = 0) :
		maxEpoch(maxEpoch),
		maxBatch(maxBatch)
	{ }

	virtual bool stop_learning(LearningState::Ptr state)
	{
		return state->epoch >= maxEpoch
			|| (maxBatch > 0 && state->batchAll >= maxBatch);
	}

	int maxEpoch;
	int maxBatch;

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(MaxEpochStopper)
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
 * Doesn't do validation/testing
 */
struct NullSchedule :
		public NoValidationSchedule, public NoTestingSchedule
{
	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(NullSchedule)
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
		return (state->epoch + 1) % validationInterval == 0;
	}

	virtual bool run_testing(LearningState::Ptr state)
	{
		if (testInterval <= 0)
			return false;
		return (state->epoch + 1) % testInterval == 0;
	}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(EpochIntervalSchedule)

protected:
	int validationInterval;
	int testInterval;
};

/**************************************
******* Observer *********
**************************************/
/**
 * Observe the network at every minibatch update
 */
// forward decl
class Network;

// For static template arg type checking only
struct ObserverBase
{
	virtual ~ObserverBase() {}
};

template<typename NetworkT>
struct Observer : public ObserverBase
{
	virtual ~Observer() {}

	void observe(std::shared_ptr<Network> net_, LearningState::Ptr state)
	{
		auto net = std::dynamic_pointer_cast<NetworkT>(net_);
		LMN_ASSERT_NULLPTR(net,
			LearningException("Observer network type mismatch"));

		this->observe_impl(net, state);
	}

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(Observer)

protected:
	virtual void observe_impl(std::shared_ptr<NetworkT>, LearningState::Ptr) = 0;
};

/**
 * Do-nothing observor
 */
struct NullObserver : public Observer<Network>
{
	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(NullObserver)

protected:
	void observe_impl(std::shared_ptr<Network>, LearningState::Ptr) { }
};

// Print every batch. Debugging
struct MinibatchObserver :
		public Observer<Network>
{
	virtual ~MinibatchObserver() {}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(MinibatchObserver)

protected:
	virtual void observe_impl(std::shared_ptr<Network>, LearningState::Ptr state)
	{
		cout << "Current minibatch: " << state->batchInEpoch
					<< " in epoch " << state->epoch << endl;
		cout << "Total training loss: " << state->trainingLoss << endl;
		cout << "Current training loss: " << state->curBatchTrainingLoss << endl;
	}
};


#endif /* LEARNING_LISTENER_H_ */
