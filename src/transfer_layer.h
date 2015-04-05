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
		outValue = sigmoid(inValue);
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

class LinearLayer : public Layer
{
public:
	LinearLayer(float _multiplier = 1.0f):
		Layer(),
		multiplier(_multiplier)
	{}

	~LinearLayer() { }

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


ostream& operator<<(ostream& os, LinearLayer& layer)
{
	os << layer.str();
	return os;
}
ostream& operator<<(ostream& os, LinearLayer&& layer)
{
	os << layer.str();
	return os;
}

ostream& operator<<(ostream& os, SigmoidLayer& layer)
{
	os << layer.str();
	return os;
}
ostream& operator<<(ostream& os, SigmoidLayer&& layer)
{
	os << layer.str();
	return os;
}

#endif /* TRANSFER_LAYER_H_ */
