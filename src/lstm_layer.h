/*
 * Eona Studio (c) 2015
 */

#ifndef LSTM_LAYER_H_
#define LSTM_LAYER_H_

#include "layer.h"
#include "parameter.h"

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
class LstmLayer : public Layer, public ParamContainer
{
public:
	LstmLayer() :
		Layer(),
		ParamContainer(LSTM_PARAM_SIZE),

		W_xi(paramValues[_W_xi]),
		W_hi(paramValues[_W_hi]),
		W_ci(paramValues[_W_ci]),
		b_i(paramValues[_b_i]),
		W_xf(paramValues[_W_xf]),
		W_hf(paramValues[_W_hf]),
		W_cf(paramValues[_W_cf]),
		b_f(paramValues[_b_f]),
		W_xc(paramValues[_W_xc]),
		W_hc(paramValues[_W_hc]),
		b_c(paramValues[_b_c]),
		W_xo(paramValues[_W_xo]),
		W_ho(paramValues[_W_ho]),
		W_co(paramValues[_W_co]),
		b_o(paramValues[_b_o]),
		h_0(paramValues[_h_0]),
		cell_0(paramValues[_cell_0]),

		W_xi_grad(paramGradients[_W_xi]),
		W_hi_grad(paramGradients[_W_hi]),
		W_ci_grad(paramGradients[_W_ci]),
		b_i_grad(paramGradients[_b_i]),
		W_xf_grad(paramGradients[_W_xf]),
		W_hf_grad(paramGradients[_W_hf]),
		W_cf_grad(paramGradients[_W_cf]),
		b_f_grad(paramGradients[_b_f]),
		W_xc_grad(paramGradients[_W_xc]),
		W_hc_grad(paramGradients[_W_hc]),
		b_c_grad(paramGradients[_b_c]),
		W_xo_grad(paramGradients[_W_xo]),
		W_ho_grad(paramGradients[_W_ho]),
		W_co_grad(paramGradients[_W_co]),
		b_o_grad(paramGradients[_b_o]),
		h_0_grad(paramGradients[_h_0]),
		cell_0_grad(paramGradients[_cell_0]),

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
		float h_last = frame() > 0 ?
				this->outValues[frame() - 1] :
				h_0;

		float cell_last = frame() > 0 ?
				this->cells[frame() - 1] :
				cell_0;

		float inputGate = gateActivator(
				W_xi * inValue + W_hi * h_last + W_ci * cell_last + b_i);
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

	vector<float> cells; // internal state history

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
		_W_xi,
		_W_hi,
		_W_ci,
		_b_i,
		_W_xf,
		_W_hf,
		_W_cf,
		_b_f,
		_W_xc,
		_W_hc,
		_b_c,
		_W_xo,
		_W_ho,
		_W_co,
		_b_o,
		_h_0,
		_cell_0,
		LSTM_PARAM_SIZE
	};

	// All parameter value aliases
	float &W_xi,
		&W_hi,
		&W_ci,
		&b_i,
		&W_xf,
		&W_hf,
		&W_cf,
		&b_f,
		&W_xc,
		&W_hc,
		&b_c,
		&W_xo,
		&W_ho,
		&W_co,
		&b_o,
		&h_0,
		&cell_0;

	// All parameter gradient aliases
	float &W_xi_grad,
		&W_hi_grad,
		&W_ci_grad,
		&b_i_grad,
		&W_xf_grad,
		&W_hf_grad,
		&W_cf_grad,
		&b_f_grad,
		&W_xc_grad,
		&W_hc_grad,
		&b_c_grad,
		&W_xo_grad,
		&W_ho_grad,
		&W_co_grad,
		&b_o_grad,
		&h_0_grad,
		&cell_0_grad;
};

#endif /* LSTM_LAYER_H_ */
