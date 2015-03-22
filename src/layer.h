/*
 * Eona Studio (c) 2015
 */


#ifndef LAYER_H_
#define LAYER_H_

class Layer
{
public:
	virtual ~Layer();

	virtual void forward(float& inValue, float& outValue) = 0;
	virtual void backward(float& inValue, float& inGradient, float& outValue, float& outGradient) = 0;
};


#endif /* LAYER_H_ */
