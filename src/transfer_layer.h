/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "math_utils.h"
#include "layer.h"

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer() :
		Layer()
	{}

	virtual ~SigmoidLayer() =default;

	void _forward(float& inValue, float& outValue)
	{
		outValue = lmn::sigmoid(inValue);
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = lmn::sigmoidGradient(outValue) * outGradient;
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

	virtual ~CosineLayer() =default;

	void _forward(float& inValue, float& outValue)
	{
		outValue = lmn::cos(inValue);
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = -lmn::sin(inValue) * outGradient;
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

	virtual ~TanhLayer() =default;

	void _forward(float& inValue, float& outValue)
	{
		outValue = lmn::tanh(inValue);
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = lmn::tanhGradient(outValue) * outGradient;
	}

	string str()
	{
		return string("[TanhLayer: \n")
				+ Layer::str() + "]";
	}
};


class ConstantLayer : public Layer
{
public:
	ConstantLayer():
		Layer()
	{}

	virtual ~ConstantLayer() =default;

	void _forward(float& inValue, float& outValue)
	{
		outValue = inValue;
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = outGradient;
	}

	string str()
	{
		return string("[ConstantLayer: \n")
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

	virtual ~ScalorLayer() =default;

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
		return string("[ScalorLayer: \n")
				+ Layer::str() + "]";
	}

private:
	float multiplier;
};

#endif /* TRANSFER_LAYER_H_ */
