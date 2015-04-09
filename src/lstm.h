/*
 * Eona Studio (c) 2015
 */

#ifndef LSTM_LAYER_H_
#define LSTM_LAYER_H_

#include "layer.h"
#include "parameter.h"
#include "composite.h"
#include "network.h"

class LstmComposite : public Composite<RecurrentNetwork>
{
public:
	LstmComposite(Layer::Ptr _inLayer) :
		Composite<RecurrentNetwork>(_inLayer)
	{ }

	virtual ~LstmComposite() =default;

	/**
	 * Composite logic goes here.
	 * Intended to work with network's "this" pointer
	 */
	virtual void manipulate(RecurrentNetwork *net)
	{
		auto forgetGate = get_layer("forget-gate");
		auto inputGate = get_layer("input-gate");
		auto cellhat = get_layer("cellhat");
		auto cell = get_layer("cell");
		auto outputGate = get_layer("output-gate");

		net->new_connection<FullConnection>(inLayer, inputGate);
		net->new_recurrent_connection<FullConnection>(outLayer, inputGate);
		net->new_recurrent_connection<FullConnection>(cell, inputGate);

		net->add_layer(inputGate);

		net->new_connection<FullConnection>(inLayer, forgetGate);
		net->new_recurrent_connection<FullConnection>(outLayer, forgetGate);
		net->new_recurrent_connection<FullConnection>(cell, forgetGate);
		net->add_layer(forgetGate);

		net->new_connection<FullConnection>(inLayer, cellhat);
		net->new_recurrent_connection<FullConnection>(outLayer, cellhat);

		net->add_layer(cellhat);

		net->new_connection<GatedConnection>(cellhat, inputGate, cell);
		net->new_recurrent_connection<GatedConnection>(cell, forgetGate, cell);

		net->add_layer(cell);

		net->new_connection<FullConnection>(inLayer, outputGate);
		net->new_recurrent_connection<FullConnection>(outLayer, outputGate);
		net->new_connection<FullConnection>(cell, outputGate);

		net->add_layer(outputGate);

		net->new_connection<GatedTanhConnection>(cell, outputGate, outLayer);

		net->add_layer(outLayer);
	}

protected:
	virtual Layer::Ptr initialize_outlayer()
	{
		return Layer::make<ConstantLayer>();
	}

	/**
	 * Will be called in constructor
	 */
	virtual void initialize_layers(
			std::unordered_map<string, Layer::Ptr>& layerMap)
	{
		layerMap["forget-gate"] = Layer::make<SigmoidLayer>();
		layerMap["input-gate"] = Layer::make<SigmoidLayer>();
		layerMap["cellhat"] = Layer::make<TanhLayer>();
		layerMap["cell"]  = Layer::make<ConstantLayer>();
		layerMap["output-gate"]  = Layer::make<SigmoidLayer>();
	}
};

/**
 * DEBUG ONLY
 */
class LstmDebugLayer : public Layer, public ParamContainer
{
public:
	LstmDebugLayer(vector<float> dummyWeights,
			vector<float> dummyPrehistory) :
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

		gateActivator(lmn::sigmoid),
		cellInputActivator(lmn::tanh),
		cellOutputActivator(lmn::tanh),
		gateActivatorGradient(lmn::sigmoidGradient),
		cellInputActivatorGradient(lmn::tanhGradient),
		cellOutputActivatorGradient(lmn::tanhGradient)
	{
		int i = 0;
		for (float* elem : { &W_xi, &W_hi, &W_ci, &W_xf, &W_hf, &W_cf, &W_xc, &W_hc, &W_xo, &W_ho, &W_co })
			*elem = dummyWeights[i ++];
		i = 0;
		// TODO add biases
		for (float* elem : { &b_i, &b_f, &b_c, &b_o })
			*elem = 0;
		i = 0;
		for (float* elem : { &h_0, &cell_0 })
			*elem = dummyPrehistory[i ++];
	}

	~LstmDebugLayer() { }

	virtual void _forward(float& inValue, float& outValue)
	{
		float h_last = frame() > 0 ?
				this->outValues[frame() - 1] :
				h_0;

		float cell_last = frame() > 0 ?
				this->cellValues[frame() - 1] :
				cell_0;

		float inputGate = gateActivator(
				W_xi * inValue + W_hi * h_last + W_ci * cell_last + b_i);

		float forgetGate = gateActivator(
				W_xf * inValue + W_hf * h_last + W_cf * cell_last + b_f);

		float cell_hat = cellInputActivator(
				W_xc * inValue + W_hc * h_last + b_c);

		float cell = inputGate * cell_hat + forgetGate * cell_last;

		vec_resize_on_demand(cellValues, frame());
		cellValues[frame()] = cell;

		float outputGate = gateActivator(
				W_xo * inValue + W_ho * h_last + W_co * cell + b_o);

		float cellOutput = cellOutputActivator(cell);

		outValue = outputGate * cellOutput;
	}

	virtual void _backward(float& outValue, float& outGradient, float& inValue, float& inGradient)
	{
		throw UnimplementedException(
				"This LSTM layer is for debugging only.\n"
				"Backprop is not supported. ");
	}

	string str()
	{
		return string("[LstmLayer (DEBUG ONLY): \n")
				+ Layer::str() + "]";
	}

	// internal state history
	vector<float> cellValues;

	function<float(float)>
		gateActivator,
		cellInputActivator,
		cellOutputActivator;

	/**
	 * Function like sigmoid's gradient can be more easily computed
	 * given the output value.
	 * We do not support gradient computation given input, because of
	 * storage concern.
	 */
	function<float(float)>
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
};

#endif /* LSTM_LAYER_H_ */
