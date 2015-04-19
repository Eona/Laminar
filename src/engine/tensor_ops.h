/*
 * Eona Studio (c) 2015
 */


#ifndef TENSOR_OPS_H_
#define TENSOR_OPS_H_

#include "tensor.h"

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
	x1.upload(Instruction(
			operand + "+" + operand, {x1.addr, x2.addr}, ans.addr));
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
	x1.upload(Instruction(
			operand + "-" + operand, {x1.addr, x2.addr}, ans.addr));
	return ans;
}

/**
 * Unary negate
 */
template<typename TensorT>
typename std::enable_if<is_tensor_base<TensorT>::value, TensorT>::type
operator-(const TensorT& x)
{
	TensorT ans(x.engine);
	string operand = tensor_class_info<TensorT>::operand;
	x.upload(Instruction("-" + operand, {x.addr}, ans.addr));
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
	x1.upload(Instruction(
			oper1 + "*" + oper2, {x1.addr, x2.addr}, ans.addr));
	return ans;
}

/**
 * Multiply by a scalor constant (float)
 * @see opcode "s*t" and "t*s"
 * @param x
 * @param scalor
 * @return
 */
Tensor operator*(const Tensor& x, float scalor)
{
	Tensor ans(x.engine);
	ans.upload(Instruction("scale", {x.addr}, ans.addr,
			OpContext<float>::make(scalor)));
	return ans;
}

Tensor operator*(float scalor, const Tensor& x)
{
	Tensor ans(x.engine);
	ans.upload(Instruction("scale", {x.addr}, ans.addr,
			OpContext<float>::make(scalor)));
	return ans;
}

/**
 * Other common deep learning operations
 */
namespace lmn
{
//typedef std::function<Tensor(const Tensor&)> TransferFunction;
typedef Tensor (*TransferFunction)(const Tensor&);

// Macro generate single tensor element-wise math operation
#define GEN_MATH_OPS(fname) \
	template<typename TensorT> \
	typename std::enable_if<is_tensor_base<TensorT>::value, TensorT>::type \
	fname(const TensorT& x) \
	{ \
		Tensor ans(x.engine); \
		x.upload(Instruction(STRINGFY(fname), {x.addr}, ans.addr)); \
		return ans; \
	}

	GEN_MATH_OPS(transpose);

	GEN_MATH_OPS(sigmoid);

	GEN_MATH_OPS(sigmoid_gradient);

	GEN_MATH_OPS(tanh);

	GEN_MATH_OPS(tanh_gradient);

	GEN_MATH_OPS(sin);

	GEN_MATH_OPS(cos);

	Tensor element_mult(const Tensor& x1, const Tensor& x2)
	{
		Tensor ans(x1.engine);
		x1.upload(Instruction(
				"element_mult", {x1.addr, x2.addr}, ans.addr));
		return ans;
	}

	// 0.5f * sum( (x1 - x2)^2 )
	Scalor square_loss(const Tensor& x1, const Tensor& x2)
	{
		Scalor ans(x1.engine);
		x1.upload(Instruction(
				"square_loss", {x1.addr, x2.addr}, ans.addr));
		return ans;
	}

	void clear(const TensorBase& x)
	{
		x.upload(Instruction("clear", {}, x.addr));
	}

	void set_value(const TensorBase& x, DimIndex idx, float val)
	{
		x.upload(Instruction("set_value", {}, x.addr,
					OpContext<DimIndex, float>::make(idx, val)));
	}

	// TODO
	void fill_rand(const Tensor& x)
	{
		x.upload(Instruction("fill_rand", {}, x.addr));
	}

	// TODO
	void fill_rand_prehistory(const Tensor& x)
	{
		x.upload(Instruction("fill_rand_prehistory", {}, x.addr));
	}

	void perturb(const Tensor& x, DimIndex idx, float eps)
	{
		x.upload(Instruction("perturb", {}, x.addr,
				OpContext<DimIndex, float>::make(idx, eps)));
	}

} // end of lmn::

#endif /* TENSOR_OPS_H_ */
