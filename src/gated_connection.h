/*
 * Eona Studio (c) 2015
 */

#ifndef GATED_CONNECTION_H_
#define GATED_CONNECTION_H_

#include "connection.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

class GatedConnection : public Connection
{
public:
	/**
	 * outLayer = gateLayer .* inLayer
	 * If used in a recurrent fashion, inLayer will be from the past while
	 * gateLayer and outLayer will both be in the current timeframe.
	 * outLayer[t] = gateLayer[t] .* inLayer[t - temporalSkip]
	 */
	GatedConnection(Layer::Ptr inLayer_, Layer::Ptr gateLayer_, Layer::Ptr outLayer_):
		Connection(inLayer_, outLayer_),
		gateLayer(gateLayer_)
	{
		LMN_ASSERT_THROW(inLayer_->dim() == gateLayer_->dim()
				&& inLayer_->dim() == outLayer_->dim(),
			ComponentException("GatedConnection inLayer, gateLayer and outLayer "
					"must have the same dimension"));
	}

	virtual ~GatedConnection() {};

	virtual void forward_impl(Tensor& inlayerOutval, Tensor& outlayerInval)
	{
		gated_forward_impl(inlayerOutval, gateLayer->out_value(out_frame()),
				// output param:
				outlayerInval);
	}

	virtual void backward_impl(Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& inlayerOutgrad)
	{
		gated_backward_impl(outlayerIngrad, inlayerOutval, gateLayer->out_value(out_frame()),
				// output params:
				inlayerOutgrad,
				gateLayer->out_gradient(
						gateLayer->is_full_gradient_history_saved() ?
								out_frame() : 0));
	}

	virtual explicit operator string() const
	{
		return "[GatedConnection]";
	}

protected:
	/*********** Subclasses should override following ***********/
	virtual void gated_forward_impl(Tensor& inlayerOutval, Tensor& gateOutval,
			// output param:
			Tensor& outlayerInval)
	{
		outlayerInval += gateOutval * inlayerOutval;
	}

	virtual void gated_backward_impl(Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& gateOutval,
			// write to output params:
			Tensor& inlayerOutgrad, Tensor& gateOutgrad)
	{
		inlayerOutgrad += gateOutval * outlayerIngrad;
		gateOutgrad += outlayerIngrad * inlayerOutval;
	}

	Layer::Ptr gateLayer;
};


/**
 * Compute outLayer = gateLayer .* nonlinear(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] .* nonlinear(inLayer[t - temporalSkip])
 *
 * Some gated gradient can be calculated more efficiently given
 * the stored output value from the history.
 */
class GatedCachedNonlinearConnection : public GatedConnection
{
public:
	GatedCachedNonlinearConnection(
			Layer::Ptr inLayer_, Layer::Ptr gateLayer_, Layer::Ptr outLayer_,
			lmn::TransferFunction nonlinear_, lmn::TransferFunction nonlinear_gradient_):
		GatedConnection(inLayer_, gateLayer_, outLayer_),
		nonlinear(nonlinear_), nonlinear_gradient(nonlinear_gradient_)
	{ }

	virtual ~GatedCachedNonlinearConnection() {};

	virtual void gated_forward_impl(Tensor& inlayerOutval, Tensor& gateOutval,
		// output param:
		Tensor& outlayerInval)
	{
		Tensor& cachedOutval = *cachedOutvals[out_frame()];

		cachedOutval = nonlinear(inlayerOutval);

		outlayerInval += lmn::element_mult(gateOutval, cachedOutval);
	}

	virtual void gated_backward_impl(Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& gateOutval,
			// write to output params:
			Tensor& inlayerOutgrad, Tensor& gateOutgrad)
	{
		Tensor& cachedOutval = *cachedOutvals[out_frame()];

		inlayerOutgrad += lmn::element_mult(
				lmn::element_mult(gateOutval, nonlinear_gradient(cachedOutval)),
				outlayerIngrad);

		gateOutgrad += lmn::element_mult(outlayerIngrad, cachedOutval);
	}

	virtual explicit operator string() const
	{
		return "[GatedCachedNonlinearConnection]";
	}

protected:
	void initialize_impl()
	{
		GatedConnection::initialize_impl();

		for (int t = 0; t < inLayer->history_length(); ++t)
		{
			cachedOutvals.push_back(Tensor::make(engine));
		}
	}

private:
	// performance acceleration ONLY
	vector<Tensor::Ptr> cachedOutvals;

	lmn::TransferFunction nonlinear;
	lmn::TransferFunction nonlinear_gradient;
};


/**
 * outLayer = gateLayer * tanh(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * tanh(inLayer[t - temporalSkip])
 */
struct GatedTanhConnection : public GatedCachedNonlinearConnection
{
	GatedTanhConnection(
			Layer::Ptr inLayer, Layer::Ptr gateLayer, Layer::Ptr outLayer):
		GatedCachedNonlinearConnection(inLayer, gateLayer, outLayer,
				lmn::tanh<Tensor>, lmn::tanh_gradient<Tensor>)
	{ }

	virtual ~GatedTanhConnection() {}
};

/**
 * outLayer = gateLayer * sigmoid(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * sigmoid(inLayer[t - temporalSkip])
 */
struct GatedSigmoidConnection : public GatedCachedNonlinearConnection
{
	GatedSigmoidConnection(
			Layer::Ptr inLayer, Layer::Ptr gateLayer, Layer::Ptr outLayer):
		GatedCachedNonlinearConnection(inLayer, gateLayer, outLayer,
				lmn::sigmoid<Tensor>, lmn::sigmoid_gradient<Tensor>)
	{ }

	virtual ~GatedSigmoidConnection() {}
};

#endif /* GATED_CONNECTION_H_ */
