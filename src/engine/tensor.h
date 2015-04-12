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
	{
		engine->upload(Instruction( "create", {}, addr));
	}

	TensorBase(EngineBase::Ptr _engine, vector<int> dim) :
		engine(_engine),
		addr(engine->alloc_dim(dim))
	{
		engine->upload(Instruction( "create_dim", {}, addr));
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

class Scalor : public TensorBase
{
public:
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
	{ }

	Tensor(EngineBase::Ptr _engine, vector<int> dim) :
		TensorBase(_engine, dim)
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

template<typename T>
struct tensor_class_info {};

template<>
struct tensor_class_info<Tensor>
{
	static constexpr const char *name = "Tensor";
	static constexpr const char *operand = "t";
};

template<>
struct tensor_class_info<Scalor>
{
	static constexpr const char *name = "Scalor";
	static constexpr const char *operand = "s";
};

/**
 * Only Tensor + Tensor or Scalor + Scalor
 */
template<typename TensorT1, typename TensorT2>
typename std::enable_if<
	is_both_tensor_bases<TensorT1, TensorT2>::value,
	TensorT1>::type
operator+(const TensorT1& x1, const TensorT2& x2)
{
	if (!std::is_same<TensorT1, TensorT2>::value)
		throw TensorException(string("operator+ type mismatch: ")
				+ tensor_class_info<TensorT1>::name + "+"
				+ tensor_class_info<TensorT2>::name + ". "
				+ "Only Tensor+Tensor or Scalor+Scalor supported.");

	using TensorT = TensorT1;
	TensorT ans(x1.engine);
	string operand = tensor_class_info<TensorT>::operand;
	x1.engine->upload(
		Instruction(operand + "+" + operand, {x1.addr, x2.addr}, ans.addr));
	return ans;
}

/**
 * Only Tensor - Tensor or Scalor - Scalor
 */
template<typename TensorT1, typename TensorT2>
typename std::enable_if<
	is_both_tensor_bases<TensorT1, TensorT2>::value,
	TensorT1>::type
operator-(const TensorT1& x1, const TensorT2& x2)
{
	if (!std::is_same<TensorT1, TensorT2>::value)
		throw TensorException(string("operator- type mismatch: ")
				+ tensor_class_info<TensorT1>::name + "-"
				+ tensor_class_info<TensorT2>::name + ". "
				+ "Only Tensor-Tensor or Scalor-Scalor supported.");

	using TensorT = TensorT1;
	TensorT ans(x1.engine);
	string operand = tensor_class_info<TensorT>::operand;
	x1.engine->upload(
		Instruction(operand + "-" + operand, {x1.addr, x2.addr}, ans.addr));
	return ans;
}

/**
 * Unary negate
 */
template<typename TensorT>
typename std::enable_if<
	is_tensor_base<TensorT>::value,
	TensorT>::type
operator-(const TensorT& x)
{
	TensorT ans(x.engine);
	string operand = tensor_class_info<TensorT>::operand;
	x.engine->upload(
		Instruction("-" + operand, {x.addr}, ans.addr));
	return ans;
}

/**
 * Multiply
 * Tensor * Tensor, Tensor * Scalor, Scalor * Tensor -> return Tensor
 * Scalor * Scalor -> Scalor
 */
template<typename TensorT1, typename TensorT2>
using select_multiply_return =
	select_type<std::is_same<TensorT1, Scalor>::value
		&& std::is_same<TensorT2, Scalor>::value,
		Scalor, Tensor>;

template<typename TensorT1, typename TensorT2>
typename std::enable_if<
	is_both_tensor_bases<TensorT1, TensorT2>::value,
	typename select_multiply_return<TensorT1, TensorT2>::type>::type
operator*(const TensorT1& x1, const TensorT2& x2)
{
	using AnsType = typename select_multiply_return<TensorT1, TensorT2>::type;
	AnsType ans(x1.engine);
	string oper1 = tensor_class_info<TensorT1>::operand;
	string oper2 = tensor_class_info<TensorT2>::operand;
	x1.engine->upload(Instruction(
			oper1 + "*" + oper2,
			{x1.addr, x2.addr}, ans.addr));
	return ans;
}

#endif /* TENSOR_H_ */
