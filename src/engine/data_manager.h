/*
 * Eona Studio (c) 2015
 */

#ifndef DATA_MANAGER_H_
#define DATA_MANAGER_H_

#include "../global_utils.h"
#include "engine.h"

class DataManagerBase
{
public:
	DataManagerBase(EngineBase::Ptr engine_) :
		engine(engine_)
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
	void upload_input(TensorBase::Ptr tensor)
	{
		engine->upload(Instruction(OP_LOAD_INPUT, {}, tensor->addr));
	}

	/**
	 * Network calls the request to fill in target Tensor
	 */
	void upload_target(TensorBase::Ptr tensor)
	{
		engine->upload(Instruction(OP_LOAD_TARGET, {}, tensor->addr));
	}

	/**
	 * Resets the input/target data stream
	 */
	virtual void start_new_epoch() = 0;

	/**
	 * If template unspecified, return EngineBase::Ptr
	 */
	template<typename EngineT = EngineBase>
	std::shared_ptr<EngineT> get_engine()
	{
		auto engine_ = std::dynamic_pointer_cast<EngineT>(this->engine);
		assert_throw_nullptr(engine_,
			NetworkException("get_engine()'s template type is incompatible"));
		return engine_;
	}

	/************************************/
	TYPEDEF_PTR(DataManagerBase);

	template<typename ManagerT, typename ...ArgT>
	static shared_ptr<ManagerT> make(ArgT&& ... args)
	{
		static_assert(std::is_base_of<DataManagerBase, ManagerT>::value,
				"make() failed: DataManager type parameter must be a subclass of DataManagerBase");

		return std::make_shared<ManagerT>(
						std::forward<ArgT>(args) ...);
	}

	/**
	 * Downcast
	 */
	template<typename ManagerT>
	static shared_ptr<ManagerT> cast(DataManagerBase::Ptr manager)
	{
		static_assert(std::is_base_of<DataManagerBase, ManagerT>::value,
				"cast() failed: DataManager type parameter must be a subclass of DataManagerBase");

		return std::dynamic_pointer_cast<ManagerT>(manager);
	}

protected:
	EngineBase::Ptr engine;
};

template<typename DataT>
class DataManager : public DataManagerBase
{
public:
	DataManager(EngineBase::Ptr engine) :
		DataManagerBase(engine)
	{
		auto specificEngine = EngineBase::cast<Engine<DataT>>(this->engine);

		specificEngine->register_normal_op(DataManagerBase::OP_LOAD_INPUT,
			[=](vector<DataT*>, DataT *write, bool is_initialized) {
				this->load_input(write, is_initialized);
			}
		);

		specificEngine->register_normal_op(DataManagerBase::OP_LOAD_TARGET,
			[=](vector<DataT*>, DataT *write, bool is_initialized) {
				this->load_target(write, is_initialized);
			}
		);
	}

	virtual void load_input(DataT *write, bool is_initialized) = 0;

	virtual void load_target(DataT *write, bool is_initialized) = 0;
};

#endif /* DATA_MANAGER_H_ */
