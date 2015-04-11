/*
 * Eona Studio (c) 2015
 */


#ifndef TENSOR_H_
#define TENSOR_H_

#include "../global_utils.h"
#include "engine.h"

struct TensorBase
{
	TensorBase(EngineBase::Ptr _engine) :
		engine(_engine)
	{
		this->addr = engine->alloc();
		engine->upload(Instruction("create", {}, addr));
	}

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
		engine->upload(Instruction("create", {}, this->addr));
		engine->upload(Instruction("copy", {other.addr}, this->addr));
	}

	// Copy assignment
	TensorBase& operator=(const TensorBase& other)
	{
		DEBUG_MSG("Copy Assign");
		engine->upload(Instruction("copy", {other.addr}, this->addr));
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

	EngineBase::Ptr engine;
	// memory address in the engine, if negative -> destroyed
	int addr;
};

struct Tensor : public TensorBase
{
	Tensor(EngineBase::Ptr _engine) :
		TensorBase(_engine)
	{ }

	virtual ~Tensor() {}

	// Copy ctor
	Tensor(const Tensor& other) :
		TensorBase(other)
	{ }

	// Copy assignment
	Tensor& operator=(const Tensor& other)
	{
		TensorBase::operator=(other);
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
};

struct Scalor : public TensorBase
{
	Scalor(EngineBase::Ptr _engine) :
		TensorBase(_engine)
	{ }

	virtual ~Scalor() {}

	// Copy ctor
	Scalor(const Scalor& other) :
		TensorBase(other)
	{ }

	// Copy assignment
	Scalor& operator=(const Scalor& other)
	{
		TensorBase::operator=(other);
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
template<typename TensorT1, typename TensorT2 >
using is_different_tensor_type =
	std::integral_constant<bool,
		std::is_base_of<TensorBase, TensorT1>::value &&
		std::is_base_of<TensorBase, TensorT2>::value &&
		!std::is_same<TensorT1, TensorT2>::value>;

/**
 * Only Tensor + Tensor or Scalor + Scalor
 */
Tensor operator+(const Tensor& x1, const Tensor& x2)
{
	Tensor ans(x1.engine);
	x1.engine->upload(Instruction("t+t", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

Scalor operator+(const Scalor& x1, const Scalor& x2)
{
	Scalor ans(x1.engine);
	x1.engine->upload(Instruction("s+s", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

template<typename TensorT1, typename TensorT2>
typename std::enable_if<
	is_different_tensor_type<TensorT1, TensorT2>::value, TensorBase>::type
operator+(const TensorT1& x1, const TensorT2& x2)
{
	throw TensorException("operator+ type mismatch: "
			"only Tensor+Tensor or Scalor+Scalor supported.");
}

/**
 * Only Tensor - Tensor or Scalor - Scalor
 */
Tensor operator-(const Tensor& x1, const Tensor& x2)
{
	Tensor ans(x1.engine);
	x1.engine->upload(Instruction("t-t", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

Scalor operator-(const Scalor& x1, const Scalor& x2)
{
	Scalor ans(x1.engine);
	x1.engine->upload(Instruction("s-s", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

/**
 * Unary negate
 */
Tensor operator-(const Tensor& x)
{
	Tensor ans(x.engine);
	x.engine->upload(Instruction("-t", {x.addr}, ans.addr));
	return ans;
}

Scalor operator-(const Scalor& x)
{
	Scalor ans(x.engine);
	x.engine->upload(Instruction("-s", {x.addr}, ans.addr));
	return ans;
}

/**
 * Multiply
 */
Tensor operator*(const Tensor& x1, const Tensor& x2)
{
	Tensor ans(x1.engine);
	x1.engine->upload(Instruction("t*t", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

Tensor operator*(const Tensor& x1, const Scalor& x2)
{
	Tensor ans(x1.engine);
	x1.engine->upload(Instruction("t*s", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

Tensor operator*(const Scalor& x1, const Tensor& x2)
{
	Tensor ans(x1.engine);
	x1.engine->upload(Instruction("s*t", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

Scalor operator*(const Scalor& x1, const Scalor& x2)
{
	Scalor ans(x1.engine);
	x1.engine->upload(Instruction("s*s", {x1.addr, x2.addr}, ans.addr));
	return ans;
}

#endif /* TENSOR_H_ */
