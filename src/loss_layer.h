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
		targetValue(1, 0.f),
		totalLoss(0.f)
	{ }

	virtual ~LossLayer() { }

	virtual float total_loss()
	{
		return totalLoss;
	}

	virtual void reset()
	{
		Layer::reset();
		totalLoss = 0;
	}

	vector<float> targetValue;

	float totalLoss;
};

TypedefPtr(LossLayer);

class SquareLossLayer : public LossLayer
{
public:
	SquareLossLayer() :
		LossLayer()
	{}

	~SquareLossLayer() { }

	virtual void _forward(float& inValue, float& outValue)
	{
		// which is loss value if the network is feedforward
		float tmp = inValue - targetValue[frame()];
		outValue = 0.5f * tmp * tmp;
		totalLoss += outValue;
	}

	virtual void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = inValue - targetValue[frame()];
	}

	string str()
	{
		return string("[SquareLossLayer: \n")
				+ Layer::str() + "]";
	}
};


#endif /* LOSS_LAYER_H_ */
