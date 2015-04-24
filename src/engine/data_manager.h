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
	 * DataManager adds the following three opcodes to your engine
	 */
	static constexpr const char *OP_LOAD_INPUT = "load_input";
	static constexpr const char *OP_LOAD_TARGET = "load_target";
	// Increment stream pointer and prepare next batch
	static constexpr const char *OP_PREPARE_NEXT_BATCH = "prepare_next_batch";

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

	/**
	 * Network calls the request to fill in target Tensor
	 */
	void upload_prepare_next_batch()
	{
		// -1 write addr for null instruction
		engine->upload(Instruction(OP_PREPARE_NEXT_BATCH, {}, -1));
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
		auto engine_ = EngineBase::cast<Engine<DataT>>(this->engine);

		engine_->register_normal_op(DataManagerBase::OP_LOAD_INPUT,
			[this](vector<DataPtr>, DataPtr write, bool is_initialized) {
				this->load_input(write, is_initialized, this->learning_stage());
			}
		);

		engine_->register_normal_op(DataManagerBase::OP_LOAD_TARGET,
			[this](vector<DataPtr>, DataPtr write, bool is_initialized) {
				this->load_target(write, is_initialized, this->learning_stage());
			}
		);

		engine_->register_normal_op(DataManagerBase::OP_PREPARE_NEXT_BATCH,
			[this](vector<DataPtr>, DataPtr write, bool is_initialized) {
				// update data manager state to be queried in LearningSession
				this->isEpochEnd =
						this->prepare_next_batch(this->learning_stage());
			}
		);
	}

	virtual void load_input(DataPtr write, bool is_initialized, LearningStage) = 0;

	// A load_target is always followed by a load_input
	virtual void load_target(DataPtr write, bool is_initialized, LearningStage) = 0;

	/**
	 * @return isEpochEnd whether we have reached the end of epoch *after* this load
	 */
	virtual bool prepare_next_batch(LearningStage) = 0;
};

#endif /* DATA_MANAGER_H_ */
