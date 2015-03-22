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
	Layer(float _inValue):
		inValue(_inValue), inGradient(0), outValue(0), outGradient(0)
	{}

	virtual ~Layer() {};

	virtual void forward()
	{
		_forward(inValue, outValue);
	}

	virtual void backward()
	{
		_backward(inValue, inGradient, outValue, outGradient);
	}

	virtual void _forward(float& inValue, float& outValue) = 0;
	virtual void _backward(float& inValue, float& inGradient, float& outValue, float& outGradient) = 0;

	virtual string str()
	{
		ostringstream os;
		os << "\tinVal=" << this->inValue
			<< "\tinGrad=" << this->inGradient
			<< "\n\toutVal=" << this->outValue
			<< "\toutGrad=" << this->outGradient;
		return os.str();
	}

	float inValue, inGradient, outValue, outGradient;
};

typedef shared_ptr<Layer> LayerPtr;

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

#endif /* LAYER_H_ */
