/*
 * Eona Studio (c) 2015
 */


#ifndef LAYER_H_
#define LAYER_H_

#include "global_utils.h"

class Layer
{
public:
	Layer(float _inValue, float _inGradient, float _outValue, float _outGradient):
		inValue(_inValue), inGradient(_inGradient), outValue(_outValue), outGradient(_outGradient)
	{}

	virtual ~Layer();

	virtual void forward(float& inValue, float& outValue) = 0;
	virtual void backward(float& inValue, float& inGradient, float& outValue, float& outGradient) = 0;

	float inValue, inGradient, outValue, outGradient;
};


#endif /* LAYER_H_ */
