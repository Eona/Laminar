/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer(Dimension dim) :
		Layer(dim)
	{}

	SigmoidLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~SigmoidLayer() {};

	void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = lmn::sigmoid(inValue);
	}

	void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
	{
		inGradient = lmn::element_mult(
				lmn::sigmoid_gradient(outValue), outGradient);
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
	CosineLayer(Dimension dim) :
		Layer(dim)
	{}

	CosineLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~CosineLayer() {};

	void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = lmn::cos(inValue);
	}

	void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
	{
		inGradient = lmn::element_mult(
				-lmn::sin(inValue), outGradient);
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
	TanhLayer(Dimension dim) :
		Layer(dim)
	{}

	TanhLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~TanhLayer() {};

	void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = lmn::tanh(inValue);
	}

	void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
	{
		inGradient = lmn::element_mult(
				lmn::tanh_gradient(outValue), outGradient);
	}

	virtual explicit operator string() const
	{
		return string("[TanhLayer: \n")
				+ Layer::operator string() + "]";
	}
};

// FIXME ScalorType cannot be a constant, discard ScalorLayer
/*class ScalorLayer : public Layer
{
public:
	ScalorLayer(Dimension dim, float multiplier_ = 1.0f):
		Layer(dim),
		multiplier(multiplier_)
	{}

	ScalorLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~ScalorLayer() {};

	void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = multiplier * inValue;
	}

	void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
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
};*/

#endif /* TRANSFER_LAYER_H_ */
