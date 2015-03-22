/*
 * Eona Studio (c) 2015
 */


#ifndef LOSS_LAYER_H_
#define LOSS_LAYER_H_

#include "layer.h"

class SquareErrorLayer : public Layer
{
public:
	float targetValue;

	SquareErrorLayer(float _inValue, float _targetValue):
		Layer(_inValue),
		targetValue(_targetValue)
	{}

	~SquareErrorLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		// which is loss value
		outValue = 0.5f * (inValue - targetValue) * (inValue - targetValue);
	}

	void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = inValue - targetValue;
	}
};


#endif /* LOSS_LAYER_H_ */
