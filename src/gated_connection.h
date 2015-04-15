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
	 * outLayer = gateLayer * inLayer
	 * If used in a recurrent fashion, inLayer will be from the past while
	 * gateLayer and outLayer will both be in the current timeframe.
	 * outLayer[t] = gateLayer[t] * inLayer[t - temporalSkip]
	 */
	GatedConnection(Layer::Ptr _inLayer, Layer::Ptr _gateLayer, Layer::Ptr _outLayer):
		Connection(_inLayer, _outLayer),
		gateLayer(_gateLayer)
	{ }

	virtual ~GatedConnection() {};

	virtual void forward_impl(Tensor& inlayerOutval, Tensor& outlayerInval)
	{
		gated_forward_impl(inlayerOutval, *gateLayer->outValues[out_frame()],
				// output param:
				outlayerInval);
	}

	virtual void backward_impl(Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& inlayerOutgrad)
	{
		gated_backward_impl(outlayerIngrad, inlayerOutval, *gateLayer->outValues[out_frame()],
				// output params:
				inlayerOutgrad,
				*gateLayer->outGradients[
						gateLayer->is_full_gradient_history_saved() ?
								out_frame() : 0]);
	}

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

	virtual explicit operator string() const
	{
		return "[GatedConnection]";
	}

protected:
	Layer::Ptr gateLayer;
};


/**
 * Compute outLayer = gateLayer * nonlinear(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * nonlinear(inLayer[t - temporalSkip])
 *
 * Some gated gradient can be calculated more efficiently given
 * the stored output value from the history.
 */
template<lmn::TransferFunction nonlinear,
		lmn::TransferFunction nonlinearGradient>
class GatedCachedNonlinearConnection : public GatedConnection
{
public:
	GatedCachedNonlinearConnection(
			Layer::Ptr _inLayer, Layer::Ptr _gateLayer, Layer::Ptr _outLayer):
		GatedConnection(_inLayer, _gateLayer, _outLayer)
	{ }

	virtual ~GatedCachedNonlinearConnection() {};

	virtual void gated_forward_impl(Tensor& inlayerOutval, Tensor& gateOutval,
		// output param:
		Tensor& outlayerInval)
	{
		int t = out_frame();
		vec_resize_on_demand(cachedOutvals, t);

		Tensor::Ptr& cached = cachedOutvals[t];
		if (!cached)
			cached = Tensor::make(engine);

		*cached = nonlinear(inlayerOutval);

		outlayerInval += gateOutval * (*cached);
	}

	virtual void gated_backward_impl(Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& gateOutval,
			// write to output params:
			Tensor& inlayerOutgrad, Tensor& gateOutgrad)
	{
		Tensor& cachedOutval = *cachedOutvals[out_frame()];

		inlayerOutgrad += gateOutval * nonlinearGradient(cachedOutval) * outlayerIngrad;
		gateOutgrad += outlayerIngrad * cachedOutval;
	}

	virtual explicit operator string() const
	{
		return "[GatedCachedNonlinearConnection]";
	}

private:
	// performance acceleration ONLY
	vector<Tensor::Ptr> cachedOutvals;
};


/**
 * outLayer = gateLayer * tanh(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * tanh(inLayer[t - temporalSkip])
 */
typedef GatedCachedNonlinearConnection<lmn::tanh, lmn::tanh_gradient>
	GatedTanhConnection;


/**
 * outLayer = gateLayer * sigmoid(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * sigmoid(inLayer[t - temporalSkip])
 */
typedef GatedCachedNonlinearConnection<lmn::sigmoid, lmn::sigmoid_gradient>
	GatedSigmoidConnection;

#endif /* GATED_CONNECTION_H_ */
