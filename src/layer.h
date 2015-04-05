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
		inGradients(2, 0.0f),
		outValues(1, 0.0f),
		outGradients(2, 0.0f)
	{ }

	virtual ~Layer() {};

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

		_backward(outValues[_frame], outGradients[1], inValues[_frame], inGradients[1]);
	}

	virtual void reset()
	{
		inValues.clear(); inValues.push_back(0);
		inGradients.clear(); inGradients.push_back(0);
		outValues.clear(); outValues.push_back(0);
		outGradients.clear(); outGradients.push_back(0);
	}

	/**
	 * Call after network does a full back_prop through all the layers
	 * Recurrent network ONLY
	 * Doing this because we are not saving the full gradient history.
	 */
	virtual void shiftBackGradientWindow()
	{
		outGradients[1] = outGradients[0];
		outGradients[0] = 0;
		inGradients[1] = inGradients[0];
		inGradients[0] = 0;
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

	vector<float> inValues,
		inGradients,
		outValues,
		outGradients;

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

/*
	static int MaxFrameInterval;

	static void setMaxFrameInterval(int _MaxFrameInterval)
	{
		MaxFrameInterval = _MaxFrameInterval;
	}
*/

private:
	// frame pointer
	int _frame = 0;
};

/**
 * Both Layer::Ptr and LayerPtr works
 */
TypedefPtr(Layer);

//int Layer::MaxFrameInterval = 2;

#endif /* LAYER_H_ */
