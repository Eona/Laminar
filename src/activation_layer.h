/*
 * Eona Studio (c) 2015
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"


/**
 * Subclasses should implement the normal forward_impl()
 * and activation_gradient()
 */
class ActivationLayer : public Layer
{
public:
	ActivationLayer(Dimension dim) :
		Layer(dim)
	{}

	ActivationLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~ActivationLayer() {};

	virtual void backward_impl(Tensor& outValue, Tensor& outGradient,
						Tensor& inValue, Tensor& inGradient)
	{
		inGradient = lmn::element_mult(
				activation_gradient(outValue, inValue), outGradient);
	}

protected:
	/*********** Subclasses should override following ***********/
	/**
	 * @param outValue
	 * @param inValue
	 * @return element-wise gradient
	 */
	virtual Tensor activation_gradient(Tensor& outValue, Tensor& inValue) = 0;
};


class SigmoidLayer : public ActivationLayer
{
public:
	SigmoidLayer(Dimension dim) :
		ActivationLayer(dim)
	{}

	SigmoidLayer(int dim) :
		ActivationLayer(dim)
	{ }

	virtual ~SigmoidLayer() {};

	void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = lmn::sigmoid(inValue);
	}

	virtual explicit operator string() const
	{
		return string("[SigmoidLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	Tensor activation_gradient(Tensor& outValue, Tensor& inValue)
	{
		return lmn::sigmoid_gradient(outValue);
	}
};

class CosineLayer : public ActivationLayer
{
public:
	CosineLayer(Dimension dim) :
		ActivationLayer(dim)
	{}

	CosineLayer(int dim) :
		ActivationLayer(dim)
	{ }

	virtual ~CosineLayer() {};

	virtual void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = lmn::cos(inValue);
	}

	virtual explicit operator string() const
	{
		return string("[CosineLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	virtual Tensor activation_gradient(Tensor& outValue, Tensor& inValue)
	{
		return -lmn::sin(inValue);
	}
};

class TanhLayer : public ActivationLayer
{
public:
	TanhLayer(Dimension dim) :
		ActivationLayer(dim)
	{}

	TanhLayer(int dim) :
		ActivationLayer(dim)
	{ }

	virtual ~TanhLayer() {};

	virtual void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = lmn::tanh(inValue);
	}

	virtual explicit operator string() const
	{
		return string("[TanhLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	virtual Tensor activation_gradient(Tensor& outValue, Tensor& inValue)
	{
		return lmn::tanh_gradient(outValue);
	}
};

/**
 * ScalarLayer is special, it overrides backward_impl directly
 * activation_gradient() will never be called
 */
class ScalarLayer : public ActivationLayer
{
public:
	ScalarLayer(Dimension dim, float multiplier_ = 1.0f):
		ActivationLayer(dim), multiplier(multiplier_)
	{}

	ScalarLayer(int dim, float multiplier_ = 1.0f) :
		ActivationLayer(dim), multiplier(multiplier_)
	{ }

	virtual ~ScalarLayer() {};

	virtual void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = multiplier * inValue;
	}

	virtual void backward_impl(Tensor& outValue, Tensor& outGradient,
						Tensor& inValue, Tensor& inGradient)
	{
		inGradient = multiplier * outGradient;
	}

	virtual explicit operator string() const
	{
		return string("[ScalarLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	Tensor activation_gradient(Tensor& outValue, Tensor& inValue)
	{
		return Tensor(engine); // doesn't do anything
	}

private:
	float multiplier;
};

#endif /* TRANSFER_LAYER_H_ */
