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
		inValue(1, 0.0f),
		inGradient(1, 0.0f),
		outValue(1, 0.0f),
		outGradient(1, 0.0f)
	{ }

	virtual ~Layer() {};

	virtual void forward(int inFrame = 0, int outFrame = 0)
	{
		if (inFrame != outFrame)
			throw UnimplementedException(
					"Layer in/out time cannot be different for now.");

		this->_frame = inFrame;
		resize_on_demand(inValue, _frame);
		resize_on_demand(outValue, _frame);
		_forward(inValue[_frame], outValue[_frame]);
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		if (outFrame != inFrame)
			throw UnimplementedException(
					"Layer in/out time cannot be different for now.");

		this->_frame = inFrame;

		resize_on_demand(outGradient, _frame);
		resize_on_demand(inGradient, _frame);
		_backward(outValue[_frame], outGradient[_frame], inValue[_frame], inGradient[_frame]);
	}

	virtual void reset()
	{
		inValue.clear(); inValue.push_back(0);
		inGradient.clear(); inGradient.push_back(0);
		outValue.clear(); outValue.push_back(0);
		outGradient.clear(); outGradient.push_back(0);
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
		os << "\tinVal=" << this->inValue
			<< "\tinGrad=" << this->inGradient
			<< "\n\toutVal=" << this->outValue
			<< "\toutGrad=" << this->outGradient;
		return os.str();
	}

	vector<float> inValue,
		inGradient,
		outValue,
		outGradient;

	/************************************/

	/**
	 * Make a polymorphic shared pointer
	 */
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

private: // frame pointer
	int _frame = 0;
};

/**
 * Both Layer::Ptr and LayerPtr works
 */
TypedefPtr(Layer);

#endif /* LAYER_H_ */
