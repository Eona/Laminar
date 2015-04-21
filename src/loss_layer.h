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
		lmn::zero_clear(*totalLoss);
	}

	TensorBase& target_value(int t)
	{
		return *this->targetValues[t];
	}

protected:
	Scalor::Ptr totalLoss;
	/**
	 * Here TensorBase::Ptr, not Tensor::Ptr because the targetValue might be Scalor:
	 * for one-hot encoding, we encapsulate 'int' class label in a Scalor (hackish)
	 */
	vector<TensorBase::Ptr> targetValues;

	/**
	 * Extend Layer::initialize.
	 * Subclasses need to extend this and initialize targetValues
	 */
	virtual void initialize_impl()
	{
		Layer::initialize_impl();
		this->totalLoss = Scalor::make(engine);
	}

	/**
	 * Used by subclasses to initialize targetValues
	 */
	template<typename TensorT>
	void init_target_value_helper()
	{
		for (int t = 0; t < history_length(); ++t)
			this->targetValues.push_back(TensorT::make(engine));
	}

	/**
	 * Get current frame target value
	 */
	template<typename TensorT>
	std::shared_ptr<TensorT> current_frame_target()
	{
		return std::dynamic_pointer_cast<TensorT>(
				this->targetValues[this->frame()]);
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
		*totalLoss += lmn::square_loss_batch(inValue, *current_frame_target<Tensor>());
	}

	virtual void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
	{
		// TODO square_loss_gradient should be implemented not as a single command, but as
		// (x1 - x2) / lmn::batch_size(x1)
		// where lmn::batch_size returns the batch dim (normally the last dim of Tensor) in a Scalor

		// inGradient = inValue - *get_cur_frame_target<Tensor>();

		inGradient = lmn::square_loss_gradient_batch(inValue, *current_frame_target<Tensor>());
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
		LossLayer::init_target_value_helper<Tensor>();
	}
};


#endif /* LOSS_LAYER_H_ */
