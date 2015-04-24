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

	virtual Scalor& loss_value() const
	{
		return *this->lossValue;
	}

	virtual Scalor::Ptr loss_value_ptr() const
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

	TensorBase& target_value(int t)
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
	Scalor::Ptr lossValue;
	/**
	 * Here TensorBase::Ptr, not Tensor::Ptr because the targetValue might be Scalor:
	 * for one-hot encoding, we encapsulate 'int' class label in a Scalor (hackish)
	 */
	vector<TensorBase::Ptr> targetValues;

	/**
	 * All subclasses must implement the following
	 * @return loss update
	 */
	virtual Scalor loss_forward_impl(Tensor& inValue, TensorBase& targetValue) = 0;

	virtual void loss_backward_impl(Tensor& inValue, TensorBase& targetValue,
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

		this->lossValue = Scalor::make(engine);
	}

	/**
	 * Used by subclasses to initialize targetValues
	 * We give a TensorT type because the target might be a Scalor (class label)
	 */
	template<typename TensorT>
	void init_target_value_helper()
	{
		for (int t = 0; t < history_length(); ++t)
			this->targetValues.push_back(TensorT::make(engine));
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
	virtual Scalor loss_forward_impl(Tensor& inValue, TensorBase& targetValue)
	{
		// we know our target value is Tensor for SquareLoss
		return lmn::square_loss(inValue, dynamic_cast<Tensor&>(targetValue));
	}

	virtual void loss_backward_impl(Tensor& inValue, TensorBase& targetValue, Tensor& inGradient)
	{
		inGradient = inValue - dynamic_cast<Tensor&>(targetValue);
	}

	virtual void initialize_impl()
	{
		LossLayer::initialize_impl();

		// For SquareLoss, our target is a Tensor
		LossLayer::init_target_value_helper<Tensor>();
	}
};


#endif /* LOSS_LAYER_H_ */
