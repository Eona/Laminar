/*
 * Eona Studio (c) 2015
 */

#ifndef LSTM_LAYER_H_
#define LSTM_LAYER_H_

#include "layer.h"
#include "parameter.h"
#include "composite.h"
#include "network.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

class LstmComposite : public Composite<RecurrentNetwork>
{
public:
	LstmComposite(Layer::Ptr inLayer_, Dimension lstmDim_) :
		Composite<RecurrentNetwork>(inLayer_),
		lstmDim(lstmDim_)
	{ }

	LstmComposite(Layer::Ptr inLayer, int lstmDim) :
		LstmComposite(inLayer, Dimension {lstmDim})
	{ }

	virtual ~LstmComposite() {};

protected:
	/**
	 * Composite logic goes here.
	 * Intended to work with network's "this" pointer
	 */
	virtual void manipulate_impl(RecurrentNetwork *net)
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

	virtual Layer::Ptr initialize_outlayer()
	{
		return Layer::make<ConstantLayer>(lstmDim);
	}

	/**
	 * Will be called in constructor
	 */
	virtual void initialize_layers(
			std::unordered_map<string, Layer::Ptr>& layerMap)
	{
		layerMap["forget-gate"] = Layer::make<SigmoidLayer>(lstmDim);
		layerMap["input-gate"] = Layer::make<SigmoidLayer>(lstmDim);
		layerMap["cellhat"] = Layer::make<TanhLayer>(lstmDim);
		layerMap["cell"]  = Layer::make<ConstantLayer>(lstmDim);
		layerMap["output-gate"]  = Layer::make<SigmoidLayer>(lstmDim);
	}

	Dimension lstmDim; // LSTM hidden unit dimension
};

/**
 * DEBUG ONLY
 */
class LstmDebugLayer : public Layer, public ParamContainer
{
public:
	LstmDebugLayer(Dimension dim,
			vector<float> dummyWeights_,
			vector<float> dummyPrehistory_) :
		Layer(dim),
		ParamContainer(LSTM_PARAM_SIZE),
		dummyWeights(dummyWeights_),
		dummyPrehistory(dummyPrehistory_),

		W_xi(param_value_ptr(_W_xi)),
		W_hi(param_value_ptr(_W_hi)),
		W_ci(param_value_ptr(_W_ci)),
		b_i(param_value_ptr(_b_i)),
		W_xf(param_value_ptr(_W_xf)),
		W_hf(param_value_ptr(_W_hf)),
		W_cf(param_value_ptr(_W_cf)),
		b_f(param_value_ptr(_b_f)),
		W_xc(param_value_ptr(_W_xc)),
		W_hc(param_value_ptr(_W_hc)),
		b_c(param_value_ptr(_b_c)),
		W_xo(param_value_ptr(_W_xo)),
		W_ho(param_value_ptr(_W_ho)),
		W_co(param_value_ptr(_W_co)),
		b_o(param_value_ptr(_b_o)),
		h_0(param_value_ptr(_h_0)),
		cell_0(param_value_ptr(_cell_0)),

		gateActivator(lmn::sigmoid<Tensor>),
		cellInputActivator(lmn::tanh<Tensor>),
		cellOutputActivator(lmn::tanh<Tensor>),
		gateActivatorGradient(lmn::sigmoid_gradient<Tensor>),
		cellInputActivatorGradient(lmn::tanh_gradient<Tensor>),
		cellOutputActivatorGradient(lmn::tanh_gradient<Tensor>)
	{ }

	LstmDebugLayer(int dim,
			vector<float> dummyWeights,
			vector<float> dummyPrehistory) :
		LstmDebugLayer(Dimension{ dim }, dummyWeights, dummyPrehistory)
	{}

	virtual ~LstmDebugLayer() { }

	virtual void forward_impl(Tensor& inValue, Tensor& outValue)
	{
		Tensor& h_last = frame() > 0 ?
				this->out_value(frame() - 1) :
				*h_0;

		Tensor& cell_last = frame() > 0 ?
				*this->cellValues[frame() - 1] :
				*cell_0;

		Tensor inputGate = gateActivator(
				*W_xi * inValue + *W_hi * h_last + *W_ci * cell_last + *b_i);

		Tensor forgetGate = gateActivator(
				*W_xf * inValue + *W_hf * h_last + *W_cf * cell_last + *b_f);

		Tensor cell_hat = cellInputActivator(
				*W_xc * inValue + *W_hc * h_last + *b_c);

		Tensor cell = inputGate * cell_hat + forgetGate * cell_last;

		*cellValues[frame()] = cell;

		Tensor outputGate = gateActivator(
				*W_xo * inValue + *W_ho * h_last + *W_co * cell + *b_o);

		Tensor cellOutput = cellOutputActivator(cell);

		outValue = outputGate * cellOutput;
	}

	virtual void backward_impl(
			Tensor& outValue, Tensor& outGradient,
			Tensor& inValue, Tensor& inGradient)
	{
		throw UnimplementedException(
				"This LSTM layer is for debugging only.\n"
				"Backprop is not supported. ");
	}

	virtual explicit operator string() const
	{
		return string("[LstmLayer (DEBUG ONLY): \n")
				+ Layer::operator string() + "]";
	}

protected:
	// internal state history
	vector<Tensor::Ptr> cellValues;
	vector<float> dummyWeights;
	vector<float> dummyPrehistory;

	virtual void initialize_impl()
	{
		Layer::initialize_impl();

		for (int t = 0; t < history_length(); ++t)
		{
			cellValues.push_back(Tensor::make(engine));
		}

		int i = 0;
		for (Tensor::Ptr* elem : { &W_xi, &W_hi, &W_ci, &W_xf, &W_hf, &W_cf, &W_xc, &W_hc, &W_xo, &W_ho, &W_co })
		{
			*elem = Tensor::make(engine, this->dim());
			lmn::set_value(**elem, {}, dummyWeights[i++]);
		}

		i = 0;
		// TODO add biases
		for (Tensor::Ptr* elem : { &b_i, &b_f, &b_c, &b_o })
		{
			*elem = Tensor::make(engine, this->dim());
			lmn::set_value(**elem, {}, 0);
		}

		i = 0;
		for (Tensor::Ptr* elem : { &h_0, &cell_0 })
		{
			*elem = Tensor::make(engine, this->dim());
			lmn::set_value(**elem, {}, dummyPrehistory[i++]);
		}
	}

	lmn::TransferFunction
		gateActivator,
		cellInputActivator,
		cellOutputActivator;

	/*
	 * Function like sigmoid's gradient can be more easily computed
	 * given the output value.
	 * We do not support gradient computation given input, because of
	 * storage concern.
	 */

	lmn::TransferFunction
		gateActivatorGradient,
		cellInputActivatorGradient,
		cellOutputActivatorGradient;

	/*
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
	Tensor::Ptr &W_xi,
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
