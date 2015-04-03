/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer.h"

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer(int timeLength = 1) :
		Layer(timeLength)
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

	string str()
	{
		return string("[SigmoidLayer: \n")
				+ Layer::str() + "]";
	}
};

class CosineLayer : public Layer
{
public:
	CosineLayer(int timeLength = 1) :
		Layer(timeLength)
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

	string str()
	{
		return string("[CosineLayer: \n")
				+ Layer::str() + "]";
	}
};

class LinearLayer : public Layer
{
public:
	LinearLayer(float _multiplier = 1.0f, int timeLength = 1):
		Layer(timeLength),
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
