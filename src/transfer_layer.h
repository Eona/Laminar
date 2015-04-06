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

		if (!is_full_gradient_history_saved())
			ParamContainer::resize(maxTemporalSkip);
	}

	// Override: if negative, use the parameter value
	virtual float& inValue(int i)
	{
		if (i >= 0)
			return Layer::inValue(i);
		else
			return paramValues[-1 - i];
	}

	// Override: if negative, use the parameter value
	virtual float& outGradient(int i)
	{
		if (i >= 0)
			return Layer::outGradient(i);
		else
			return paramGradients[-1 - i];
	}

	virtual void reset()
	{
		Layer::reset();
		ParamContainer::resetGradients();
	}
};

class SigmoidLayer : public SimpleTemporalLayer
{
public:
	SigmoidLayer() :
		SimpleTemporalLayer()
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

class CosineLayer : public SimpleTemporalLayer
{
public:
	CosineLayer() :
		SimpleTemporalLayer()
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

class LinearLayer : public SimpleTemporalLayer
{
public:
	LinearLayer(float _multiplier = 1.0f):
		SimpleTemporalLayer(),
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
