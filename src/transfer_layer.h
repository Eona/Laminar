/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer.h"

/**
 * Holds "prehistory" parameters h[-1], h[-2] ... h[-maxTemporalSkip]
 */
class SimpleTemporalLayer : public Layer, public ParamContainer
{
public:
	SimpleTemporalLayer() :
		Layer(),
		ParamContainer(maxTemporalSkip)
	{ }

	virtual void set_max_temporal_skip(int maxTemporalSkip)
	{
		Layer::set_max_temporal_skip(maxTemporalSkip);
		ParamContainer::resize(maxTemporalSkip);
	}
};

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

#endif /* TRANSFER_LAYER_H_ */
