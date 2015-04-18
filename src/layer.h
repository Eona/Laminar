/*
 * Eona Studio (c) 2015
 */


#ifndef LAYER_H_
#define LAYER_H_

#include "global_utils.h"
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

	Dimension dim()
	{
		return this->dim_;
	}

	/**
	 * Maximum temporal skip, allows a hidden layer to link (skip) to its
	 * future at +skip timestep. The most typical RNN has maxTemporalSkip = 1.
	 * Default = 0 for feed-forward network.
	 * If the value is UNLIMITED_TEMPORAL_SKIP, we save the full gradient history
	 * Must be called before initialize()
	 */
	void init_max_temporal_skip(int maxTemporalSkip)
	{
		Component::check_uninitialized("init_max_temporal_skip", "Layer");

		this->maxTemporalSkip = maxTemporalSkip;
	}

	int max_temporal_skip()
	{
		return this->maxTemporalSkip;
	}

	bool is_full_gradient_history_saved()
	{
		return this->maxTemporalSkip == UNLIMITED_TEMPORAL_SKIP;
	}

	/**
	 * Input sequence length, default = 1
	 */
	void init_history_length(int historyLength)
	{
		Component::check_uninitialized("init_history_length", "Layer");

		this->historyLength = historyLength;
	}

	int history_length()
	{
		return this->historyLength;
	}

	// FIXME do we need batch size? I think null Tensors can figure out the inflow dims
/*	void init_batch_size(int batchSize)
	{
		assert_throw(!this->is_initialized,
			ComponentException("init_batch_size() must be called before initialize()"));

		this->batchSize = batchSize;
	}*/

	virtual void forward(int inFrame = 0, int outFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);

		this->frame_ = inFrame;

		forward_impl(*inValues[frame_], *outValues[frame_]);
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);

		this->frame_ = inFrame;
		int relativeFrame = is_full_gradient_history_saved() ? frame_ : 0;

		backward_impl(*outValues[frame_],
				*outGradients[relativeFrame],
				*inValues[frame_],
				*inGradients[relativeFrame]);
	}

	virtual void zero_clear()
	{
		for (int i = 0; i < this->historyLength; ++i)
		{
			lmn::clear(*inValues[i]);
			lmn::clear(*outValues[i]);
		}

		int gradientHistoryLength =
			is_full_gradient_history_saved() ? historyLength : maxTemporalSkip + 1;

		for (int i = 0; i < gradientHistoryLength; ++i)
		{
			lmn::clear(*inGradients[i]);
			lmn::clear(*outGradients[i]);
		}
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
		ostringstream os;
		os << "\tinVal=" << this->inValues
			<< "\tinGrad=" << this->inGradients
			<< "\n\toutVal=" << this->outValues
			<< "\toutGrad=" << this->outGradients;
		return os.str();
	}

	/************************************/
	TYPEDEF_PTR(Layer);

	template<typename LayerT, typename ...ArgT>
	static Layer::Ptr make(ArgT&& ... args)
	{
		static_assert(std::is_base_of<Layer, LayerT>::value,
				"make() failed: type parameter must be a subclass of Layer");

		return std::static_pointer_cast<Layer>(
				std::make_shared<LayerT>(
						std::forward<ArgT>(args) ...));
	}

	/**
	 * Down cast LayerPtr to a specific layer type
	 */
	template<typename LayerT>
	static shared_ptr<LayerT> cast(Layer::Ptr layer)
	{
		static_assert(std::is_base_of<Layer, LayerT>::value,
				"cast() failed: type parameter must be a subclass of Layer");

		return std::dynamic_pointer_cast<LayerT>(layer);
	}

protected:
	/**
	 * Implement Component::initialize
	 */
	virtual void initialize_impl()
	{
		for (int t = 0; t < historyLength; ++t)
		{
			inValues.push_back(Tensor::make(engine));
			outValues.push_back(Tensor::make(engine));
		}

		int gradientHistoryLength =
			is_full_gradient_history_saved() ? historyLength : maxTemporalSkip + 1;

		for (int t = 0; t < gradientHistoryLength; ++t)
		{
			inGradients.push_back(Tensor::make(engine));
			outGradients.push_back(Tensor::make(engine));
		}
	}

	// Shift the gradient window
	// FIXME memory is not being saved, still alloc a lot of memory
	void shift_back_vector(vector<Tensor::Ptr>& grad)
	{
//		grad.insert(grad.begin(), 0);
//		grad.erase(grad.end() - 1);
		grad.push_back(Tensor::make(engine));
		grad.erase(grad.begin());
	}

	void check_frame_consistency(int inFrame, int outFrame)
	{
		assert_throw(inFrame == outFrame,
			UnimplementedException(
				"Layer in/out time cannot be different."));
	}

private:
	Dimension dim_;

	// frame pointer
	int frame_ = 0;

protected:
	int historyLength;

	// Max temporal skip. negative to save full gradient history
	int maxTemporalSkip;

public: // FIXME no public!
	vector<Tensor::Ptr> inValues,
				inGradients,
				outValues,
				outGradients;

public:
	enum : int {
		UNLIMITED_TEMPORAL_SKIP = -1
	};
};

/**
 * Both Layer::Ptr and LayerPtr works
 */
TYPEDEF_PTR_EXTERNAL(Layer);


// Special layer that has inValue=outValue, inGradient=outGradient aliases
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

	void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		// FIXME WARNING must multiply by 1.f, otherwise free pointer error!!!
		outValue = 1.f*inValue;
	}

	void backward_impl(Tensor& outValue, Tensor& outGradient,
			Tensor& inValue, Tensor& inGradient)
	{
		inGradient = 1.f*outGradient;
	}

	virtual explicit operator string() const
	{
		return string("[ConstantLayer: \n")
				+ Layer::operator string() + "]";
	}
};

#endif /* LAYER_H_ */
