/*
 * Eona Studio (c) 2015
 */


#ifndef LAYER_H_
#define LAYER_H_

#include "global_utils.h"
#include "math_utils.h"

class Layer
{
public:
	Layer(float _inValue):
		inValue(_inValue), inGradient(0), outValue(0), outGradient(0)
	{}

	virtual ~Layer() {};

	void forward()
	{
		_forward(inValue, outValue);
	}

	void backward()
	{
		_backward(inValue, inGradient, outValue, outGradient);
	}

	virtual void _forward(float& inValue, float& outValue) = 0;
	virtual void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient) = 0;

	float inValue, inGradient, outValue, outGradient;
};


#endif /* LAYER_H_ */
