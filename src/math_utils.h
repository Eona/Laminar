/*
 * Eona Studio (c) 2015
 */


#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#include "global_utils.h"

// FIXME for prototype only
namespace lmn
{
	inline float transpose(float x) { return x; }

	inline float softmax(float x) { return x; }

	inline float sigmoid(float x)
	{
		return 1.0f / (1.0f + exp(-x));
	}

	inline float tanh(float x)
	{
		return std::tanh(x);
	}

	inline float sin(float x)
	{
		return std::sin(x);
	}

	inline float cos(float x)
	{
		return std::cos(x);
	}

	inline float sigmoidGradient(float outValue)
	{
		return outValue * (1.f - outValue);
	}

	inline float tanhGradient(float outValue)
	{
		return 1.f - outValue * outValue;
	}

	typedef function<float(float)> TransferFunction;
} // end of namespace

#endif /* MATH_UTILS_H_ */
