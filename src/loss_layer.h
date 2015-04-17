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

	virtual Scalor::Ptr total_loss()
	{
		return this->totalLoss;
	}

	virtual void zero_clear()
	{
		Layer::zero_clear();
		lmn::clear(*totalLoss);
	}

	// FIXME no public!
	/**
	 * Here TensorBase::Ptr, not Tensor::Ptr because the targetValue might be Scalor:
	 * for one-hot encoding, we encapsulate 'int' class label in a Scalor (hackish)
	 */
	vector<TensorBase::Ptr> targetValue;

protected:
	Scalor::Ptr totalLoss;

	/**
	 * Extend Layer::initialize.
	 * Subclasses need to extend this and initialize targetValue
	 */
	virtual void initialize_impl()
	{
		Layer::initialize_impl();
		totalLoss = Scalor::make(engine);
	}

	// FIXME template check
	/**
	 * Get current frame target value
	 */
	template<typename TensorT>
	std::shared_ptr<TensorT> get_cur_frame_target()
	{
		return std::dynamic_pointer_cast<TensorT>(
				this->targetValue[this->frame()]);
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

	// TODO all LossLayer shouldn't have outValue Tensor
	virtual void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		// which is loss value if the network is feedforward
		*totalLoss += lmn::square_loss(inValue, *get_cur_frame_target<Tensor>());
	}

	virtual void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
	{
		inGradient = inValue - *get_cur_frame_target<Tensor>();
	}

	virtual explicit operator string() const
	{
		return string("[SquareLossLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	virtual void initialize_impl()
	{
		LossLayer::initialize_impl();
		for (int t = 0; t < historyLength; ++t)
		{
			targetValue.push_back(Tensor::make(engine));
		}
	}
};


#endif /* LOSS_LAYER_H_ */
