/*
 * Eona Studio (c) 2015
 */


#ifndef LAYER_H_
#define LAYER_H_

#include "connection.h"
#include "component.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

class Layer : public Component
{
public:
	Layer(Dimension dim) :
		dim_(dim),
		historyLength(1),
		maxTemporalSkip(0)
	{ }

	Layer(int dim) :
		Layer(Dimension{ dim })
	{ }

	virtual ~Layer() {};

	Dimension dim() const
	{
		return this->dim_;
	}

	enum : int {
		UNLIMITED_TEMPORAL_SKIP = -1
	};
	/**
	 * Maximum temporal skip, allows a hidden layer to link (skip) to its
	 * future at +skip timestep. The most typical RNN has maxTemporalSkip = 1.
	 * Default = 0 for feed-forward network.
	 * If the value is UNLIMITED_TEMPORAL_SKIP, we save the full gradient history
	 * Must be called before initialize()
	 */
	void init_max_temporal_skip(int maxTemporalSkip)
	{
		Component::initGuard.assert_before_initialize("init_max_temporal_skip", "Layer");

		this->maxTemporalSkip = maxTemporalSkip;
	}

	int max_temporal_skip() const
	{
		return this->maxTemporalSkip;
	}

	bool is_full_gradient_history_saved() const
	{
		return this->maxTemporalSkip == UNLIMITED_TEMPORAL_SKIP;
	}

	/**
	 * Input sequence length, default = 1
	 */
	void init_history_length(int historyLength)
	{
		Component::initGuard.assert_before_initialize("init_history_length", "Layer");

		this->historyLength = historyLength;
	}

	int history_length() const
	{
		return this->historyLength;
	}

	/**
	 * @return if full gradient history is saved, return historyLength,
	 * otherwise return maxTemporalSkip + 1
	 */
	int gradient_history_length() const
	{
		return is_full_gradient_history_saved() ?
				historyLength : maxTemporalSkip + 1;
	}

	/*********** Getter for in/out value/gradient ***********/
	virtual Tensor& in_value(int t) const
	{
		return *this->inValues[t];
	}
	virtual Tensor::Ptr in_value_ptr(int t) const
	{
		return this->inValues[t];
	}

	virtual Tensor& in_gradient(int t) const
	{
		return *this->inGradients[t];
	}
	virtual Tensor::Ptr in_gradient_ptr(int t) const
	{
		return this->inGradients[t];
	}

	virtual Tensor& out_value(int t) const
	{
		return *this->outValues[t];
	}
	virtual Tensor::Ptr out_value_ptr(int t) const
	{
		return this->outValues[t];
	}

	virtual Tensor& out_gradient(int t) const
	{
		return *this->outGradients[t];
	}
	virtual Tensor::Ptr out_gradient_ptr(int t) const
	{
		return this->outGradients[t];
	}

	virtual void forward(int inFrame = 0, int outFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);

		this->frame_ = inFrame;

		forward_impl(in_value(frame_), out_value(frame_));
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);

		this->frame_ = inFrame;
		int relativeFrame = is_full_gradient_history_saved() ? frame_ : 0;

		backward_impl(out_value(frame_),
				out_gradient(relativeFrame),
				in_value(frame_),
				in_gradient(relativeFrame));
	}

	virtual void zero_clear()
	{
		zero_clear_invalue();
		zero_clear_outvalue();
		zero_clear_ingradient();
		zero_clear_outgradient();
	}

	/**
	 * Call after network does a full back_prop through all the layers
	 * ONLY if recurrent network AND maxTemporalSkip != UNLIMITED_TEMPORAL_SKIP
	 * Do this when we are not saving the full gradient history.
	 * [11, 22, 33] => [22, 33, 0] // lower index is more recent frame
	 */
	virtual void shift_back_gradient_window()
	{
		if (!is_full_gradient_history_saved())
		{
			shift_back_vector(outGradients);
			shift_back_vector(inGradients);
		}
	}

	virtual void forward_impl(
			Tensor& inValue, Tensor& outValue) = 0;
	virtual void backward_impl(
			Tensor& inValue, Tensor& inGradient,
			Tensor& outValue, Tensor& outGradient) = 0;

	// current time frame set by forward() and backward()
	int frame()
	{
		return this->frame_;
	}

	virtual explicit operator string() const
	{
		std::ostringstream os;
		os << "\tinVal=" << this->inValues
			<< "\tinGrad=" << this->inGradients
			<< "\n\toutVal=" << this->outValues
			<< "\toutGrad=" << this->outGradients;
		return os.str();
	}

	/************************************/
	TYPEDEF_PTR(Layer);

	GEN_MAKE_STATIC_MEMBER(Layer)

	/**
	 * Down cast LayerPtr to a specific layer type
	 */
	GEN_DOWN_CAST_STATIC_MEMBER(Layer)

protected:
	/**
	 * Implement Component::initialize
	 */
	virtual void initialize_impl()
	{
		initialize_impl_invalue();
		initialize_impl_outvalue();
		initialize_impl_ingradient();
		initialize_impl_outgradient();
	}

	/**
	 * Helpers for special subclasses that don't want to initialize all fields
	 */
	void initialize_impl_invalue()
	{
		for (int t = 0; t < historyLength; ++t)
			inValues.push_back(Tensor::make(engine));
	}
	void initialize_impl_outvalue()
	{
		for (int t = 0; t < historyLength; ++t)
			outValues.push_back(Tensor::make(engine));
	}
	void initialize_impl_ingradient()
	{
		for (int t = 0; t < gradient_history_length(); ++t)
			inGradients.push_back(Tensor::make(engine));
	}
	void initialize_impl_outgradient()
	{
		for (int t = 0; t < gradient_history_length(); ++t)
			outGradients.push_back(Tensor::make(engine));
	}

	/**
	 * Helpers for special subclasses that don't need to zero clear all fields
	 */
	void zero_clear_invalue()
	{
		for (int t = 0; t < historyLength; ++t)
			lmn::zero_clear(in_value(t));
	}
	void zero_clear_outvalue()
	{
		for (int t = 0; t < historyLength; ++t)
			lmn::zero_clear(out_value(t));
	}
	void zero_clear_ingradient()
	{
		for (int t = 0; t < gradient_history_length(); ++t)
			lmn::zero_clear(in_gradient(t));
	}
	void zero_clear_outgradient()
	{
		for (int t = 0; t < gradient_history_length(); ++t)
			lmn::zero_clear(out_gradient(t));
	}


	// Shift the gradient window
	// FIXME memory is not being saved, still alloc a lot of memory
	void shift_back_vector(vector<Tensor::Ptr>& grad)
	{
		if (!grad.empty())
		{
			grad.push_back(Tensor::make(engine));
			grad.erase(grad.begin());
		}
	}

	void check_frame_consistency(int inFrame, int outFrame)
	{
		LMN_ASSERT_THROW(inFrame == outFrame,
				UnimplementedException("Layer in/out time cannot be different."));
	}

private:
	Dimension dim_;

	// frame pointer
	int frame_ = 0;

	int historyLength;

	// Max temporal skip. negative to save full gradient history
	int maxTemporalSkip;

	vector<Tensor::Ptr> inValues,
				inGradients,
				outValues,
				outGradients;
};

/**
 * Both Layer::Ptr and LayerPtr works
 */
TYPEDEF_PTR_EXTERNAL(Layer);


/**
 * Special layer that has inValue=>outValue, inGradient=>outGradient aliases
 * They point to the same physical memory
 */
class ConstantLayer : public Layer
{
public:
	ConstantLayer(Dimension dim):
		Layer(dim)
	{}

	ConstantLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~ConstantLayer() {};

	// NOTE override accessor methods [in/out]_[value/gradient]
	// so that out_value() actually points to in_value()
	// trick the callers so that they are agnostic of the alias
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


	virtual void zero_clear()
	{
		Layer::zero_clear_invalue();
		Layer::zero_clear_ingradient();
	}

	void forward_impl(Tensor& inValue, Tensor& outValue) {}

	void backward_impl(Tensor& outValue, Tensor& outGradient,
			Tensor& inValue, Tensor& inGradient) {}

	virtual explicit operator string() const
	{
		return string("[ConstantLayer: \n")
				+ Layer::operator string() + "]";
	}

protected:
	/**
	 * inValues == outValues, inGradients == outGradients
	 * We only initialize one alias
	 */
	virtual void initialize_impl()
	{
		Layer::initialize_impl_invalue();
		Layer::initialize_impl_ingradient();
	}
};

#endif /* LAYER_H_ */
