/*
 * Eona Studio (c) 2015
 */


#ifndef TENSOR_H_
#define TENSOR_H_

#include "../global_utils.h"
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

	// Copy ctor
	TensorBase(const TensorBase& other) :
		engine(other.engine),
		addr(engine->alloc())
	{
		DEBUG_MSG("Copy Ctor");
	}

	// Copy assignment
	TensorBase& operator=(const TensorBase& other)
	{
		DEBUG_MSG("Copy Assign");
		return *this;
	}

	// Move ctor
	TensorBase(TensorBase&& other) :
		engine(other.engine),
		addr(other.addr)
	{
		DEBUG_MSG("Move Ctor");
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

	void register_engine(EngineBase::Ptr engine)
	{
		this->engine = engine;
	}

	EngineBase::Ptr engine;
	// memory address in the engine, if negative -> destroyed
	int addr;
};

class Scalor : public TensorBase
{
public:
	// If default constructed, must call register_engine() later
	Scalor() {}

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
};

class Tensor : public TensorBase
{
public:
	Tensor(EngineBase::Ptr _engine) :
		TensorBase(_engine)
	{
		engine->upload(Instruction( "create_null_t", {}, addr));
	}

	Tensor(EngineBase::Ptr _engine, vector<int> dim) :
		TensorBase(_engine)
	{
		engine->set_dim(this->addr, dim);
		engine->upload(Instruction( "create", {}, addr));
	}

	virtual ~Tensor() {}

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
};


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
