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

	void _forward(float& inValue, float& outValue)
	{
		// which is loss value if the network is feedforward
		outValue = 0.5f * (inValue - targetValue[time_pt()]) * (inValue - targetValue[time_pt()]);
		totalLoss += outValue;
	}

	void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = inValue - targetValue[time_pt()];
	}

	string str()
	{
		return string("[SquareLossLayer: \n")
				+ Layer::str() + "]";
	}
};


#endif /* LOSS_LAYER_H_ */
