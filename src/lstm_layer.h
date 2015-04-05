/*
 * Eona Studio (c) 2015
 */

#ifndef LSTM_LAYER_H_
#define LSTM_LAYER_H_

#include "layer.h"

float sigmoidGradient(float inValue, float outValue)
{
	return outValue * (1.f - outValue);
}

float tanhGradient(float inValue, float outValue)
{
	return 1 - outValue * outValue;
}

/**
 * http://deeplearning.net/tutorial/lstm.html
 * LSTM is inherently recurrent. No need to add explicit recurrent connection.
 */
class LstmLayer : public Layer, public ParameterContainer
{
public:
	LstmLayer() :
		Layer(),
		ParameterContainer(LSTM_NUMBER_OF_PARAMS),
		gateActivator(lmn::sigmoid),
		cellInputActivator(lmn::tanh),
		cellOutputActivator(lmn::tanh),
		gateActivatorGradient(sigmoidGradient),
		cellInputActivatorGradient(tanhGradient),
		cellOutputActivatorGradient(tanhGradient)
	{}

	~LstmLayer() { }

	void _forward(float& inValue, float& outValue)
	{
		outValue = 1.0f / (1.0f + exp(-inValue));
	}

	void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		inGradient = outValue * (1.0f - outValue) * outGradient;
	}

	string str()
	{
		return string("[LstmLayer: \n")
				+ Layer::str() + "]";
	}

	function<float(float)>
		gateActivator,
		cellInputActivator,
		cellOutputActivator;

	/**
	 * Function like sigmoid's gradient can be more easily computed
	 * given the output value
	 */
	function<float(float, float)>
		gateActivatorGradient,
		cellInputActivatorGradient,
		cellOutputActivatorGradient;

	/**
	 * Parameter index positions
	 * x: inValue[frame]
	 * i: input gate
	 * f: forget gate
	 * h: outValue[frame - 1] // previous time hidden layer value
	 * c: state cell
	 * o: output gate
	 * b: bias
	 */
	enum {
		W_xi,
		W_hi,
		W_ci,
		b_i,
		W_xf,
		W_hf,
		W_cf,
		b_f,
		W_xc,
		W_hc,
		b_c,
		W_xo,
		W_ho,
		W_co,
		b_o,
		LSTM_NUMBER_OF_PARAMS
	};
};

#endif /* LSTM_LAYER_H_ */
