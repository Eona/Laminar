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
	Layer(float _inValue, float _inGradient, float _outValue, float _outGradient):
		inValue(_inValue), inGradient(_inGradient), outValue(_outValue), outGradient(_outGradient)
	{}

	virtual ~Layer();

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
