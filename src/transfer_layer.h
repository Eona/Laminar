/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer.h"

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer(float _inValue):
		Layer(_inValue)
	{}

	~SigmoidLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = 1.0f / (1.0f + exp(-inValue));
	}

	void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = outValue * (1.0f - outValue) * outGradient;
	}
};

class CosineLayer : public Layer
{
public:
	CosineLayer(float _inValue):
		Layer(_inValue)
	{}

	~CosineLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = cos(inValue);
	}

	void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = -sin(inValue) * outGradient;
	}
};

class LinearLayer : public Layer
{
public:
	LinearLayer(float _inValue, float _multiplier = 1.0f):
		Layer(_inValue),
		multiplier(_multiplier)
	{}

	~LinearLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = multiplier * inValue;
	}

	void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = multiplier * outGradient;
	}

private:
	float multiplier;
};


#endif /* TRANSFER_LAYER_H_ */
