/*
 * Eona Studio (c) 2015
 */


#ifndef LOSS_LAYER_H_
#define LOSS_LAYER_H_

#include "layer.h"

class LossLayer : public Layer
{
public:
	LossLayer() {}

	virtual ~LossLayer() { }

	float targetValue = 0;
};

TypedefPtr(LossLayer);

class SquareLossLayer : public LossLayer
{
public:
	SquareLossLayer() {}

	~SquareLossLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		// which is loss value
		outValue = 0.5f * (inValue - targetValue) * (inValue - targetValue);
	}

	void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = inValue - targetValue;
	}

	string str()
	{
		return string("[SquareLossLayer: \n")
				+ Layer::str() + "]";
	}
};


#endif /* LOSS_LAYER_H_ */
