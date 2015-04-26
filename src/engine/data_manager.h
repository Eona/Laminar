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
		learnPhase(LearningPhase::Training),
		isEndOfEpoch({ false, false, false })
	{ }

	virtual ~DataManagerBase()
	{}

	/**
	 * DataManager adds the following three opcodes to your engine
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

	LearningPhase learning_phase() const
	{
		return this->learnPhase;
	}

	void set_learning_phase(LearningPhase newPhase)
	{
		this->learnPhase = newPhase;
	}

	void prepare_next_batch()
	{
		this->isEndOfEpoch[enum2integral(learnPhase)] =
				this->prepare_next_batch_impl(this->learnPhase);
	}

	/**
	 * Reset the data stream to the beginning for a new epoch
	 */
	void reset_epoch()
	{
		this->reset_epoch_impl(learnPhase);
		this->isEndOfEpoch[enum2integral(learnPhase)] = false;
	}

	/**
	 * Set by load_input()
	 * @return true if this epoch ends after this batch.
	 */
	bool is_end_of_epoch() const
	{
		return this->isEndOfEpoch[enum2integral(learnPhase)];
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
	 * Derived should implement this
	 * @return isEpochEnd whether we have reached the end of epoch *after* this load
	 */
	virtual bool prepare_next_batch_impl(LearningPhase) = 0;

	/**
	 * Derived should implement this
	 * Reset the data stream to the beginning for a new epoch
	 */
	virtual void reset_epoch_impl(LearningPhase) = 0;

private:
	LearningPhase learnPhase;
	/**
	 * If input/target stream has ended (current epoch finishes)
	 */
	std::array<bool, LEARNING_PHASE_N> isEndOfEpoch;

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
			[this](vector<DataPtr>, DataPtr write, bool is_initialized)
			{
				LMN_ASSERT_THROW(!this->is_end_of_epoch(),
					DataException("load_input failure because end of epoch reached."));

				this->load_input(write, is_initialized, this->learning_phase());
			}
		);

		engine_->register_normal_op(DataManagerBase::OP_LOAD_TARGET,
			[this](vector<DataPtr>, DataPtr write, bool is_initialized)
			{
				LMN_ASSERT_THROW(!this->is_end_of_epoch(),
					DataException("load_target failure because end of epoch reached."));

				this->load_target(write, is_initialized, this->learning_phase());
			}
		);
	}

	virtual void load_input(DataPtr write, bool is_initialized, LearningPhase) = 0;

	// A load_target is always followed by a load_input
	virtual void load_target(DataPtr write, bool is_initialized, LearningPhase) = 0;
};

#endif /* DATA_MANAGER_H_ */
