/*
 * Eona Studio (c) 2015
 */


#ifndef LAYER_H_
#define LAYER_H_

#include "global_utils.h"
#include "math_utils.h"
#include "connection.h"
#include "component.h"

class Layer : public Component
{
public:
	Layer() :
		inValues(1, 0.0f),
		inGradients(maxTemporalSkip + 1, 0.0f),
		outValues(1, 0.0f),
		outGradients(maxTemporalSkip + 1, 0.0f)
	{ }

	virtual ~Layer() {};

	/**
	 * Maximum temporal skip, allows a hidden layer to link (skip) to its
	 * future at +skip timestep. The most typical RNN has maxTemporalSkip = 1.
	 * Default to 0 for feed-forward network.
	 * If the value is UNLIMITED_TEMPORAL_SKIP, we save the full gradient history
	 * If you change maxTemporalSkip, the gradient vector will be extended or shrinked.
	 * Need to manually reset the gradient to ensure consistency.
	 */
	void set_max_temporal_skip(int maxTemporalSkip)
	{
		if (maxTemporalSkip != UNLIMITED_TEMPORAL_SKIP)
		{
			inGradients.resize(maxTemporalSkip + 1);
			outGradients.resize(maxTemporalSkip + 1);
		}
		else
		{
			inGradients.clear();
			outGradients.clear();
		}
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

	virtual void forward(int inFrame = 0, int outFrame = 0)
	{
		if (inFrame != outFrame)
			throw UnimplementedException(
					"Layer in/out time cannot be different for now.");

		this->_frame = inFrame;
		resize_on_demand(inValues, _frame);
		resize_on_demand(outValues, _frame);
		_forward(inValues[_frame], outValues[_frame]);
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		if (outFrame != inFrame)
			throw UnimplementedException(
					"Layer in/out time cannot be different for now.");

		this->_frame = inFrame;

		if (maxTemporalSkip == UNLIMITED_TEMPORAL_SKIP)
		{
			resize_on_demand(inGradients, _frame);
			resize_on_demand(outGradients, _frame);
		}

		_backward(outValues[_frame],
				is_full_gradient_history_saved() ?
					outGradients[_frame] :
					vec_at(outGradients, -1),
				inValues[_frame],
				is_full_gradient_history_saved() ?
					inGradients[_frame] :
					vec_at(inGradients, -1));
	}

	virtual void reset()
	{
		std::fill(inValues.begin(), inValues.end(), 0);
		std::fill(outValues.begin(), outValues.end(), 0);
		std::fill(inGradients.begin(), inGradients.end(), 0);
		std::fill(outGradients.begin(), outGradients.end(), 0);
	}

	/**
	 * Call after network does a full back_prop through all the layers
	 * ONLY if recurrent network AND maxTemporalSkip != UNLIMITED_TEMPORAL_SKIP
	 * Do this when we are not saving the full gradient history.
	 * [11, 22, 33] => [0, 11, 22]
	 */
	virtual void shiftBackGradientWindow()
	{
		if (!is_full_gradient_history_saved())
		{
			shiftBackVector(outGradients);
			shiftBackVector(inGradients);
		}
	}

	virtual void _forward(float& inValue, float& outValue) = 0;
	virtual void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient) = 0;

	// current time frame set by forward() and backward()
	int frame()
	{
		return this->_frame;
	}

	virtual string str()
	{
		ostringstream os;
		os << "\tinVal=" << this->inValues
			<< "\tinGrad=" << this->inGradients
			<< "\n\toutVal=" << this->outValues
			<< "\toutGrad=" << this->outGradients;
		return os.str();
	}

	/************************************/
	typedef shared_ptr<Layer> Ptr;

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
	// Shift the gradient window
	static void shiftBackVector(vector<float>& grad)
	{
		grad.insert(grad.begin(), 0);
		grad.erase(grad.end() - 1);
	}

	// Max temporal skip. negative to save full gradient history
	int maxTemporalSkip = 0;

private:
	// frame pointer
	int _frame = 0;

public:
	vector<float> inValues,
		inGradients,
		outValues,
		outGradients;

	enum : int {
		UNLIMITED_TEMPORAL_SKIP = -1
	};
};

/**
 * Both Layer::Ptr and LayerPtr works
 */
TypedefPtr(Layer);

//int Layer::MaxFrameInterval = 2;

#endif /* LAYER_H_ */
