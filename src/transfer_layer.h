/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer.h"

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer() :
		Layer()
	{}

	~SigmoidLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = lmn::sigmoid(inValue);
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = outValue * (1.0f - outValue) * outGradient;
	}

	string str()
	{
		return string("[SigmoidLayer: \n")
				+ Layer::str() + "]";
	}
};

class CosineLayer : public Layer
{
public:
	CosineLayer() :
		Layer()
	{}

	~CosineLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = cos(inValue);
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = -sin(inValue) * outGradient;
	}

	string str()
	{
		return string("[CosineLayer: \n")
				+ Layer::str() + "]";
	}
};

class TanhLayer : public Layer
{
public:
	TanhLayer() :
		Layer()
	{}

	~TanhLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = tanh(inValue);
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = (1.f - outValue * outValue) * outGradient;
	}

	string str()
	{
		return string("[TanhLayer: \n")
				+ Layer::str() + "]";
	}
};

class ScalorLayer : public Layer
{
public:
	ScalorLayer(float _multiplier = 1.0f):
		Layer(),
		multiplier(_multiplier)
	{}

	~ScalorLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = multiplier * inValue;
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = multiplier * outGradient;
	}

	string str()
	{
		return string("[LinearLayer: \n")
				+ Layer::str() + "]";
	}

private:
	float multiplier;
};

#endif /* TRANSFER_LAYER_H_ */
