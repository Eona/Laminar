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
	LossLayer(vector<int> dim) :
		Layer(dim)
	{ }

	virtual ~LossLayer() {};

	virtual Scalor total_loss()
	{
		return totalLoss;
	}

	virtual void reset()
	{
		Layer::reset();
		// TODO clear totalLoss
	}

	// FIXME no public!
	vector<Tensor::Ptr> targetValue;

	Scalor::Ptr totalLoss;

protected:
	/**
	 * Extend Layer::initialize
	 */
	virtual void initialize_impl()
	{
		Layer::initialize_impl();

		totalLoss = Scalor::make(engine);

		for (int t = 0; t < historyLength; ++t)
		{
			targetValue.push_back(Tensor::make(engine));
		}
	}

};

TYPEDEF_PTR(LossLayer);

class SquareLossLayer : public LossLayer
{
public:
	SquareLossLayer(vector<int> dim) :
		LossLayer(dim)
	{}

	virtual ~SquareLossLayer() {};

	// TODO all LossLayer shouldn't have outValue Tensor
	virtual void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		// which is loss value if the network is feedforward
		totalLoss += lmn::square_loss(inValue, targetValue[frame()]);
	}

	virtual void backward_impl(Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)
	{
		inGradient = inValue - targetValue[frame()];
	}

	virtual explicit operator string() const
	{
		return string("[SquareLossLayer: \n")
				+ Layer::operator string() + "]";
	}
};


#endif /* LOSS_LAYER_H_ */
