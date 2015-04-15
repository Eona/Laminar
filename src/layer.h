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
	Layer(vector<int> dim) :
		dim_(dim),
		historyLength(1),
		maxTemporalSkip(0)
	{ }

	Layer(int dim) :
		Layer(vector<int>{ dim })
	{ }

	virtual ~Layer() {};

	vector<int> dim()
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
		assert_throw(!this->is_initialized,
			ComponentException("init_max_temporal_skip() must be called before initialize()"));

		this->maxTemporalSkip = maxTemporalSkip;
	}

	int get_max_temporal_skip()
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
		assert_throw(!this->is_initialized,
			ComponentException("init_history_length() must be called before initialize()"));

		this->historyLength = historyLength;
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

	// TODO set all to zero matrices for more convenient gradient check
/*	virtual void zero_clear()
	{
		std::fill(inValues.begin(), inValues.end(), 0);
		std::fill(outValues.begin(), outValues.end(), 0);
		std::fill(inGradients.begin(), inGradients.end(), 0);
		std::fill(outGradients.begin(), outGradients.end(), 0);
	}*/

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
		return static_cast<Layer::Ptr>(
				std::make_shared<LayerT>(
						std::forward<ArgT>(args) ...));
	}

	/**
	 * Down cast LayerPtr to a specific layer type
	 */
	template<typename LayerT>
	static shared_ptr<LayerT> cast(Layer::Ptr layer)
	{
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

	/**
	 * Implement Component::reset
	 */
	virtual void reset_impl()
	{
		inValues.clear();
		outValues.clear();
		inGradients.clear();
		outGradients.clear();
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
	vector<int> dim_;

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
	ConstantLayer(vector<int> dim):
		Layer(dim)
	{}

	ConstantLayer(int dim) :
		Layer(dim)
	{ }

	virtual ~ConstantLayer() {};

	void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		outValue = inValue;
	}

	void backward_impl(Tensor& outValue, Tensor& outGradient,
			Tensor& inValue, Tensor& inGradient)
	{
		inGradient = outGradient;
	}

	virtual explicit operator string() const
	{
		return string("[ConstantLayer: \n")
				+ Layer::operator string() + "]";
	}
};

#endif /* LAYER_H_ */
