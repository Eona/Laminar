/*
 * Eona Studio (c) 2015
 */

#ifndef DATA_MANAGER_H_
#define DATA_MANAGER_H_

#include "../utils/global_utils.h"
#include "../learning_listener.h"
#include "engine.h"

class DataManagerBase
{
public:
	DataManagerBase(EngineBase::Ptr engine_) :
		engine(engine_),
		learnStage(LearningStage::Training)
	{ }

	virtual ~DataManagerBase()
	{}

	/**
	 * DataManager adds the following two opcodes to your engine
	 */
	static constexpr const char *OP_LOAD_INPUT = "load_input";
	static constexpr const char *OP_LOAD_TARGET = "load_target";

	/**
	 * Network calls the request to fill in input Tensor
	 */
	void upload_input(const TensorBase& tensor)
	{
		engine->upload(Instruction(OP_LOAD_INPUT, {}, tensor.addr));
	}

	/**
	 * Network calls the request to fill in target Tensor
	 */
	void upload_target(const TensorBase& tensor)
	{
		engine->upload(Instruction(OP_LOAD_TARGET, {}, tensor.addr));
	}

	LearningStage learning_stage() const
	{
		return this->learnStage;
	}

	void set_learning_stage(LearningStage newStage)
	{
		this->learnStage = newStage;
	}

	/**
	 * Reset the data stream to the beginning for a new epoch
	 */
	virtual void reset_epoch(LearningStage) = 0;

	/**
	 * Set by load_input()
	 * @return true if this epoch ends after this batch.
	 */
	bool is_epoch_end() const
	{
		return this->isEpochEnd;
	}

	virtual Dimension input_dim() const = 0;

	virtual Dimension target_dim() const = 0;

	virtual int batch_size() const = 0;

	/**
	 * If template unspecified, return EngineBase::Ptr
	 */
	template<typename EngineT = EngineBase>
	std::shared_ptr<EngineT> get_engine()
	{
		auto engine_ = std::dynamic_pointer_cast<EngineT>(this->engine);
		LMN_ASSERT_NULLPTR(engine_,
			NetworkException("get_engine()'s template type is incompatible"));
		return engine_;
	}

	/************************************/
	TYPEDEF_PTR(DataManagerBase);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(DataManagerBase)

	/**
	 * Downcast
	 */
	GEN_DOWN_CAST_STATIC_MEMBER(DataManagerBase)

protected:
	EngineBase::Ptr engine;

	/**
	 * If input/target stream has ended (current epoch finishes)
	 */
	bool isEpochEnd = false;

private:
	LearningStage learnStage;
};

template<typename DataT>
class DataManager : public DataManagerBase
{
public:
	typedef std::shared_ptr<DataT> DataPtr;

	DataManager(EngineBase::Ptr engine) :
		DataManagerBase(engine)
	{
		auto specificEngine = EngineBase::cast<Engine<DataT>>(this->engine);

		specificEngine->register_normal_op(DataManagerBase::OP_LOAD_INPUT,
			[=](vector<DataPtr>, DataPtr write, bool is_initialized) {
				this->isEpochEnd =
					this->load_input(write, is_initialized, this->learning_stage());
			}
		);

		specificEngine->register_normal_op(DataManagerBase::OP_LOAD_TARGET,
			[=](vector<DataPtr>, DataPtr write, bool is_initialized) {
				this->load_target(write, is_initialized, this->learning_stage());
			}
		);
	}

	/**
	 * @param write
	 * @param is_initialized
	 * @param stage
	 * @return isEpochEnd whether we have reached the end of epoch *after* this load
	 */
	virtual bool load_input(DataPtr write, bool is_initialized, LearningStage) = 0;

	// A load_target is always followed by a load_input
	virtual void load_target(DataPtr write, bool is_initialized, LearningStage) = 0;
};

#endif /* DATA_MANAGER_H_ */
