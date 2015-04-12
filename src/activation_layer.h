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

	virtual ~SigmoidLayer() {};

	void forward_impl(float& inValue, float& outValue)
	{
		outValue = lmn::sigmoid(inValue);
	}

	void backward_impl(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = lmn::sigmoidGradient(outValue) * outGradient;
	}

	virtual explicit operator string() const
	{
		return string("[SigmoidLayer: \n")
				+ Layer::operator string() + "]";
	}
};

class CosineLayer : public Layer
{
public:
	CosineLayer() :
		Layer()
	{}

	virtual ~CosineLayer() {};

	void forward_impl(float& inValue, float& outValue)
	{
		outValue = lmn::cos(inValue);
	}

	void backward_impl(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = -lmn::sin(inValue) * outGradient;
	}

	virtual explicit operator string() const
	{
		return string("[CosineLayer: \n")
				+ Layer::operator string() + "]";
	}
};

class TanhLayer : public Layer
{
public:
	TanhLayer() :
		Layer()
	{}

	virtual ~TanhLayer() {};

	void forward_impl(float& inValue, float& outValue)
	{
		outValue = lmn::tanh(inValue);
	}

	void backward_impl(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = lmn::tanhGradient(outValue) * outGradient;
	}

	virtual explicit operator string() const
	{
		return string("[TanhLayer: \n")
				+ Layer::operator string() + "]";
	}
};

class ScalorLayer : public Layer
{
public:
	ScalorLayer(float _multiplier = 1.0f):
		Layer(),
		multiplier(_multiplier)
	{}

	virtual ~ScalorLayer() {};

	void forward_impl(float& inValue, float& outValue)
	{
		outValue = multiplier * inValue;
	}

	void backward_impl(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = multiplier * outGradient;
	}

	virtual explicit operator string() const
	{
		return string("[ScalorLayer: \n")
				+ Layer::operator string() + "]";
	}

private:
	float multiplier;
};

#endif /* TRANSFER_LAYER_H_ */
