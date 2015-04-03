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

	virtual void forward(int inTime = 0, int outTime = 0)
	{
		if (inTime != outTime)
			throw UnimplementedException(
					"Layer in/out time cannot be different for now.");

		this->timept = inTime;
		resize_on_demand(inValue, timept);
		resize_on_demand(outValue, timept);
		_forward(inValue[timept], outValue[timept]);
	}

	virtual void backward(int outTime = 0, int inTime = 0)
	{
		if (outTime != inTime)
			throw UnimplementedException(
					"Layer in/out time cannot be different for now.");

		this->timept = inTime;

		resize_on_demand(outGradient, timept);
		resize_on_demand(inGradient, timept);
		_backward(outValue[timept], outGradient[timept], inValue[timept], inGradient[timept]);
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

	// current time set by forward() and backward()
	int time_pt() { return this->timept; }

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

private: // time pointer
	int timept = 0;
};

TypedefPtr(Layer);

/**
 * Make a polymorphic shared pointer
 */
template<typename LayerT, typename ...ArgT>
LayerPtr make_layer(ArgT&& ... args)
{
	return static_cast<LayerPtr>(
			std::make_shared<LayerT>(
					std::forward<ArgT>(args) ...));
}

/**
 * Down cast LayerPtr to a specific layer type
 */
template<typename LayerT>
shared_ptr<LayerT> cast_layer(LayerPtr layer)
{
	return std::dynamic_pointer_cast<LayerT>(layer);
}

#endif /* LAYER_H_ */
