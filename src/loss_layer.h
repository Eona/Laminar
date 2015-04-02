/*
 * Eona Studio (c) 2015
 */


#ifndef LOSS_LAYER_H_
#define LOSS_LAYER_H_

#include "layer.h"

class LossLayer : public Layer
{
public:
	LossLayer() :
		Layer(),
		targetValue(1, 0.0f)
	{ }

	virtual ~LossLayer() { }

	vector<float> targetValue;
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
		outValue = 0.5f * (inValue - targetValue[time]) * (inValue - targetValue[time]);
	}

	void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = inValue - targetValue[time];
	}

	string str()
	{
		return string("[SquareLossLayer: \n")
				+ Layer::str() + "]";
	}
};


#endif /* LOSS_LAYER_H_ */
