/*
 * Eona Studio (c) 2015
 */


#ifndef LOSS_LAYER_H_
#define LOSS_LAYER_H_

#include "layer.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

class LossLayer : public Layer
{
public:
	LossLayer(Dimension dim) :
		Layer(dim)
	{ }

	LossLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~LossLayer() {};

	virtual Scalar& loss_value() const
	{
		return *this->lossValue;
	}

	virtual Scalar::Ptr loss_value_ptr() const
	{
		return this->lossValue;
	}

	virtual void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		*lossValue += loss_forward_impl(inValue, *targetValues[frame()]);
	}

	virtual void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
	{
		loss_backward_impl(inValue, *targetValues[frame()], inGradient);
	}

	virtual void zero_clear()
	{
		// outValue and outGradients are never used or initialized
		Layer::zero_clear_invalue();
		Layer::zero_clear_ingradient();

		lmn::zero_clear(*lossValue);
	}

	Tensor& target_value(int t)
	{
		return *this->targetValues[t];
	}

	// Fake out_value and out_gradient, loss layer needs neither
	virtual Tensor& out_value(int t) const
	{
		return in_value(t);
	}
	virtual Tensor::Ptr out_value_ptr(int t) const
	{
		return in_value_ptr(t);
	}

	virtual Tensor& out_gradient(int t) const
	{
		return in_gradient(t);
	}
	virtual Tensor::Ptr out_gradient_ptr(int t) const
	{
		return in_gradient_ptr(t);
	}

protected:
	Scalar::Ptr lossValue;
	/**
	 * int class labels can be faked as a 1-by-batchSize tensor
	 */
	vector<Tensor::Ptr> targetValues;

	/**
	 * All subclasses must implement the following
	 * @return loss update
	 */
	virtual Scalar loss_forward_impl(Tensor& inValue, Tensor& targetValue) = 0;

	virtual void loss_backward_impl(Tensor& inValue, Tensor& targetValue,
									// output parameter:
									Tensor& inGradient) = 0;

	/**
	 * Extend Layer::initialize.
	 * Subclasses need to extend this and initialize targetValues
	 */
	virtual void initialize_impl()
	{
		// Loss layer doesn't need outValue or outGradient
		Layer::initialize_impl_invalue();
		Layer::initialize_impl_ingradient();

		for (int t = 0; t < history_length(); ++t)
			this->targetValues.push_back(Tensor::make(engine));

		this->lossValue = Scalar::make(engine);
	}
};

TYPEDEF_PTR_EXTERNAL(LossLayer);

class SquareLossLayer : public LossLayer
{
public:
	SquareLossLayer(Dimension dim) :
		LossLayer(dim)
	{}

	SquareLossLayer(int dim) :
		LossLayer(dim)
	{ }

	virtual ~SquareLossLayer() {};

	virtual explicit operator string() const
	{
		return string("[SquareLossLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	virtual Scalar loss_forward_impl(Tensor& inValue, Tensor& targetValue)
	{
		return lmn::square_loss(inValue, targetValue);
	}

	virtual void loss_backward_impl(Tensor& inValue, Tensor& targetValue, Tensor& inGradient)
	{
		inGradient = inValue - targetValue;
	}
};

/**
 * With integer labelled classes
 */
class LabelSoftmaxEntropyLayer : public LossLayer
{
public:
	LabelSoftmaxEntropyLayer(Dimension dim) :
		LossLayer(dim)
	{}

	LabelSoftmaxEntropyLayer(int dim) :
		LossLayer(dim)
	{ }

	virtual ~LabelSoftmaxEntropyLayer() {};

	virtual explicit operator string() const
	{
		return string("[LabelSoftmaxLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	vector<Tensor::Ptr> cachedSoftmax;

	virtual Scalar loss_forward_impl(Tensor& inValue, Tensor& targetValue)
	{
		*cachedSoftmax[frame()] = lmn::softmax(inValue);

		// our target is an integer class label
		return lmn::label_entropy_loss(inValue, targetValue);
	}

	virtual void loss_backward_impl(Tensor& inValue, Tensor& targetValue, Tensor& inGradient)
	{
		inGradient = lmn::label_softmax_entropy_gradient(
				// y - t
				// where t is a sparse vector with a single '1' at the correct label
				*cachedSoftmax[frame()], targetValue);
	}

	virtual void initialize_impl()
	{
		LossLayer::initialize_impl();

		for (int t = 0; t < history_length(); ++t)
			this->cachedSoftmax.push_back(Tensor::make(engine));
	}
};


#endif /* LOSS_LAYER_H_ */
