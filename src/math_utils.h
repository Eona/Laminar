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

} // end of namespace

#endif /* MATH_UTILS_H_ */
