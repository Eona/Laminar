/*
 * Eona Studio (c) 2015
 */

#ifndef DATA_MANAGER_H_
#define DATA_MANAGER_H_

#include "../utils/global_utils.h"
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
	 * Reset the data stream to the beginning for a new epoch
	 */
	virtual void start_new_epoch() = 0;

	/**
	 * Return current epoch number.
	 * Should update at the start of every new epoch
	 */
	virtual int current_epoch() = 0;

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

	template<typename ManagerT, typename ...ArgT>
	static std::shared_ptr<ManagerT> make(ArgT&& ... args)
	{
		LMN_STATIC_ASSERT((std::is_base_of<DataManagerBase, ManagerT>::value),
			"make() failed: DataManager type parameter must be a subclass of DataManagerBase");

		return std::make_shared<ManagerT>(
						std::forward<ArgT>(args) ...);
	}

	/**
	 * Downcast
	 */
	GEN_DOWN_CAST_STATIC_MEMBER(DataManagerBase)

protected:
	EngineBase::Ptr engine;
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
				this->load_input(write, is_initialized);
			}
		);

		specificEngine->register_normal_op(DataManagerBase::OP_LOAD_TARGET,
			[=](vector<DataPtr>, DataPtr write, bool is_initialized) {
				this->load_target(write, is_initialized);
			}
		);
	}

	virtual void load_input(DataPtr write, bool is_initialized) = 0;

	virtual void load_target(DataPtr write, bool is_initialized) = 0;
};

#endif /* DATA_MANAGER_H_ */
