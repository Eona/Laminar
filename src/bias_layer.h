/*
 * Eona Studio (c) 2015
 */

#ifndef BIAS_LAYER_H_
#define BIAS_LAYER_H_

#include "layer.h"

/**
 * Always pass to outValue [1, 1, 1... ]
 * The number of ones equal batch size
 */
class BiasLayer : public Layer
{
public:
	/**
	 * Bias layer always have dimension 1
	 */
	BiasLayer():
		Layer( { 1 } )
	{}

	virtual ~BiasLayer() {};

	TYPEDEF_PTR(BiasLayer);

	void init_batch_size(int batchSize)
	{
		Component::initGuard.assert_before_initialize("init_batch_size", "BiasLayer");

		this->batchSize = batchSize;
	}

	int batch_size() const
	{
		return this->batchSize;
	}

	// NOTE override accessor methods [in/out]_[value/gradient]
	// so that in/out_value() always points to a constant [1, 1, 1 ...]
	// in/out_gradient() always points to a dummy Tensor
	virtual Tensor& in_value(int t) const
	{
		return *this->biasActivation;
	}
	virtual Tensor::Ptr in_value_ptr(int t) const
	{
		return this->biasActivation;
	}
	virtual Tensor& out_value(int t) const
	{
		return *this->biasActivation;
	}
	virtual Tensor::Ptr out_value_ptr(int t) const
	{
		return this->biasActivation;
	}

	virtual Tensor& in_gradient(int t) const
	{
		return *this->placeholder;
	}
	virtual Tensor::Ptr in_gradient_ptr(int t) const
	{
		return this->placeholder;
	}
	virtual Tensor& out_gradient(int t) const
	{
		return *this->placeholder;
	}
	virtual Tensor::Ptr out_gradient_ptr(int t) const
	{
		return this->placeholder;
	}


	// Do nothing
	void zero_clear() { }

	void forward_impl(Tensor& inValue, Tensor& outValue) {}

	void backward_impl(Tensor& outValue, Tensor& outGradient,
			Tensor& inValue, Tensor& inGradient) {}

	virtual explicit operator string() const
	{
		return string("[ConstantLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:

	virtual void initialize_impl()
	{
		LMN_ASSERT_THROW(batchSize != 0,
			ComponentException("BiasLayer batch size is uninitialized. "
					"Call init_batch_size() before initialization."));

		this->biasActivation = Tensor::make(engine, Dimension{ 1, batchSize });

		lmn::fill_element<float>(*biasActivation,
				[](Dimension)->float { return 1.f; });

		this->placeholder = Tensor::make(engine);
	}

	int batchSize = 0;

	// A constant value [1, 1, 1...] #1's == batchSize
	Tensor::Ptr biasActivation;

	// A placeholder for in/outGradient
	Tensor::Ptr placeholder;
};


#endif /* BIAS_LAYER_H_ */
