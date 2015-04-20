/*
 * Eona Studio (c) 2015
 */


#ifndef TENSOR_H_
#define TENSOR_H_

#include "engine.h"

class TensorBase
{
public:
	TensorBase(EngineBase::Ptr _engine) :
		engine(_engine),
		addr(engine->alloc())
	{ }

	virtual ~TensorBase()
	{
		if (this->addr != -1)
		{
			engine->upload(Instruction("destroy", {}, addr));
			this->addr = -1;
		}
	}

	inline void upload(Instruction instr) const
	{
		this->engine->upload(instr);
	}

	// Copy ctor
	TensorBase(const TensorBase& other) :
		engine(other.engine),
		addr(engine->alloc())
	{
	}

	// Copy assignment
	TensorBase& operator=(const TensorBase& other)
	{
		return *this;
	}

	// Move ctor
	TensorBase(TensorBase&& other) :
		engine(other.engine),
		addr(other.addr)
	{
		other.addr = -1;
	}

	// Move assignment
/*	TensorBase& operator=(TensorBase&& other)
	{
		DEBUG_MSG("Move Assign");
		this->addr = other.addr;
		other.addr = -1;
		return *this;
	}*/

	TYPEDEF_PTR(TensorBase);

	EngineBase::Ptr engine;
	// memory address in the engine, if negative -> destroyed
	int addr;
};

class Scalor : public TensorBase
{
public:
	Scalor(EngineBase::Ptr _engine) :
		TensorBase(_engine)
	{
		engine->upload(Instruction( "create_null_s", {}, addr));
	}

	virtual ~Scalor() {}

	// Copy ctor
	Scalor(const Scalor& other) :
		TensorBase(other)
	{
		engine->upload(Instruction("create_null_s", {}, this->addr));
		engine->upload(Instruction("s=s", {other.addr}, this->addr));
	}

	// Copy assignment
	Scalor& operator=(const Scalor& other)
	{
		TensorBase::operator=(other);
		engine->upload(Instruction("s=s", {other.addr}, this->addr));
		return *this;
	}

	// Move ctor
	Scalor(Scalor&& other) :
		TensorBase(std::move(other))
	{ }

	// Move assignment
/*	Scalor& operator=(Scalor&& other)
	{
		TensorBase::operator=(std::move(other));
		return *this;
	}*/

	Scalor& operator+=(const Scalor& other)
	{
		engine->upload(Instruction("s+s", {this->addr, other.addr}, this->addr));
		return *this;
	}

	Scalor& operator-=(const Scalor& other)
	{
		engine->upload(Instruction("s-s", {this->addr, other.addr}, this->addr));
		return *this;
	}

	Scalor& operator*=(const Scalor& other)
	{
		engine->upload(Instruction("s*s", {this->addr, other.addr}, this->addr));
		return *this;
	}

	/************************************/
	TYPEDEF_PTR(Scalor);

	template<typename ...ArgT>
	static Scalor::Ptr make(ArgT&& ... args)
	{
		return std::make_shared<Scalor>(
						std::forward<ArgT>(args) ...);
	}
};

class Tensor : public TensorBase
{
public:
	Tensor(EngineBase::Ptr engine) :
		TensorBase(engine)
	{
		engine->upload(Instruction( "create_null_t", {}, addr));
	}

	Tensor(EngineBase::Ptr engine, Dimension dim) :
		TensorBase(engine), dim_(dim)
	{
		engine->upload(Instruction("create", {}, addr,
				OpContext<Dimension>::make(dim)));
	}

	virtual ~Tensor() {}

	Dimension dim() const
	{
		assert_throw(!dim_.empty(),
			TensorException("null created, cannot query dimension."));

		return this->dim_;
	}

	// Copy ctor
	Tensor(const Tensor& other) :
		TensorBase(other)
	{
		engine->upload(Instruction("create_null_t", {}, this->addr));
		engine->upload(Instruction("t=t", {other.addr}, this->addr));
	}

	// Copy assignment
	Tensor& operator=(const Tensor& other)
	{
		TensorBase::operator=(other);
		engine->upload(Instruction("t=t", {other.addr}, this->addr));
		return *this;
	}

	// Move ctor
	Tensor(Tensor&& other) :
		TensorBase(std::move(other))
	{ }

	// Move assignment
/*	Tensor& operator=(Tensor&& other)
	{
		TensorBase::operator=(std::move(other));
		return *this;
	}*/

	Tensor& operator+=(const Tensor& other)
	{
		engine->upload(Instruction("t+t", {this->addr, other.addr}, this->addr));
		return *this;
	}

	Tensor& operator-=(const Tensor& other)
	{
		engine->upload(Instruction("t-t", {this->addr, other.addr}, this->addr));
		return *this;
	}

	Tensor& operator*=(const Tensor& other)
	{
		engine->upload(Instruction("t*t", {this->addr, other.addr}, this->addr));
		return *this;
	}

	Tensor& operator*=(const Scalor& other)
	{
		engine->upload(Instruction("t*s", {this->addr, other.addr}, this->addr));
		return *this;
	}

	Tensor& operator*=(float scalor)
	{
		engine->upload(Instruction("scale", {this->addr}, this->addr,
				OpContext<float>::make(scalor)));
		return *this;
	}

	TYPEDEF_PTR(Tensor);

	template<typename ...ArgT>
	static Tensor::Ptr make(ArgT&& ... args)
	{
		return std::make_shared<Tensor>(
                    std::forward<ArgT>(args) ...);
	}

private:
	Dimension dim_;
};

/**
 * WARNING: Define Engine<DataT> member function here to
 * avoid mutual header inclusion with tensor.h
 */
template<typename DataT>
inline Engine<DataT>::DataPtr Engine<DataT>::read_memory(TensorBase::Ptr tensorPtr)
{
	int addr = tensorPtr->addr;
	assert_throw(this->memoryPool.is_initialized(addr),
		EngineException("MemoryPool[] address not initialized."));
	return this->memoryPool[addr];
}

template<typename DataT>
inline Engine<DataT>::DataPtr Engine<DataT>::read_memory(const TensorBase& tensorPtr)
{
	int addr = tensorPtr.addr;
	assert_throw(this->memoryPool.is_initialized(addr),
		EngineException("MemoryPool[] address not initialized."));
	return this->memoryPool[tensorPtr.addr];
}


/*
// same as below. Type alias is more idiomatic in c++11
template<typename TensorT1, typename TensorT2, typename JudgeT = void>
struct is_different_tensor_type : std::false_type {};

template<typename TensorT1, typename TensorT2 >
struct is_different_tensor_type<TensorT1, TensorT2,
	typename std::enable_if<
		std::is_base_of<TensorBase, TensorT1>::value &&
		std::is_base_of<TensorBase, TensorT2>::value &&
		!std::is_same<TensorT1, TensorT2>::value
	>::type
>: std::true_type {};
*/
template<typename TensorT>
using is_tensor_base =
	std::integral_constant<bool,
		std::is_base_of<TensorBase, TensorT>::value>;

template<typename TensorT1, typename TensorT2>
using is_both_tensor_bases =
	std::integral_constant<bool,
		is_tensor_base<TensorT1>::value
		&& is_tensor_base<TensorT2>::value>;

template<typename TensorT1, typename TensorT2 >
using is_different_tensor_type =
	std::integral_constant<bool,
		is_both_tensor_bases<TensorT1, TensorT2>::value
		&& !std::is_same<TensorT1, TensorT2>::value>;

#endif /* TENSOR_H_ */
