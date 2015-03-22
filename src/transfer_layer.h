/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer.h"

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer(float _inValue, float _inGradient, float _outValue, float _outGradient):
		Layer(_inValue, _inGradient, _outValue, _outGradient)
	{}

	~SigmoidLayer() { }

	void forward(float& inValue, float& outValue)
	{
		outValue = 1.0f / (1.0f + exp(inValue));
	}

	void backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = outValue * (1.0f - outValue) * outGradient;
	}
};

class CosineLayer : public Layer
{
public:
	CosineLayer(float _inValue, float _inGradient, float _outValue, float _outGradient):
		Layer(_inValue, _inGradient, _outValue, _outGradient)
	{}

	~CosineLayer() { }

	void forward(float& inValue, float& outValue)
	{
		outValue = cos(inValue);
	}

	void backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = -sin(inValue) * outGradient;
	}
};

class LinearLayer : public Layer
{
public:
	LinearLayer(float _inValue, float _inGradient, float _outValue, float _outGradient,
			float _multiplier):
		Layer(_inValue, _inGradient, _outValue, _outGradient),
		multiplier(_multiplier)
	{}

	~LinearLayer() { }

	void forward(float& inValue, float& outValue)
	{
		outValue = multiplier * inValue;
	}

	void backward(float& inValue, float& inGradient, float& outValue, float& outGradient)
	{
		inGradient = multiplier * outGradient;
	}

private:
	float multiplier;
};


#endif /* TRANSFER_LAYER_H_ */
