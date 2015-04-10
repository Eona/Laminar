/*
 * Eona Studio (c) 2015
 */

#ifndef GATED_CONNECTION_H_
#define GATED_CONNECTION_H_

#include "connection.h"
#include "math_utils.h"

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

	virtual ~GatedConnection() =default;

	virtual void forward_impl(float inlayerOutval, float& outlayerInval)
	{
		gated_forward_impl(inlayerOutval, gateLayer->outValues[out_frame()],
				// output param:
				outlayerInval);
	}

	virtual void backward_impl(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad)
	{
		gated_backward_impl(outlayerIngrad, inlayerOutval, gateLayer->outValues[out_frame()],
				// output params:
				inlayerOutgrad,
				gateLayer->outGradients[
						gateLayer->is_full_gradient_history_saved() ?
								out_frame() : 0]);
	}

	/*********** Subclasses should override following ***********/
	virtual void gated_forward_impl(float& inlayerOutval, float& gateOutval,
			// output param:
			float& outlayerInval)
	{
		outlayerInval += gateOutval * inlayerOutval;
	}

	virtual void gated_backward_impl(float& outlayerIngrad, float& inlayerOutval, float& gateOutval,
			// write to output params:
			float& inlayerOutgrad, float& gateOutgrad)
	{
		inlayerOutgrad += gateOutval * outlayerIngrad;
		gateOutgrad += outlayerIngrad * inlayerOutval;
	}

	string str()
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

	virtual ~GatedCachedNonlinearConnection() =default;

	virtual void gated_forward_impl(float& inlayerOutval, float& gateOutval,
		// output param:
		float& outlayerInval)
	{
		int t = out_frame();
		vec_resize_on_demand(cachedOutvals, t);

		cachedOutvals[t] = nonlinear(inlayerOutval);

		outlayerInval += gateOutval * cachedOutvals[t];
	}

	virtual void gated_backward_impl(float& outlayerIngrad, float& inlayerOutval, float& gateOutval,
			// write to output params:
			float& inlayerOutgrad, float& gateOutgrad)
	{
		float cachedOutval = cachedOutvals[out_frame()];

		inlayerOutgrad += gateOutval * nonlinearGradient(cachedOutval) * outlayerIngrad;
		gateOutgrad += outlayerIngrad * cachedOutval;
	}

	string str()
	{
		return "[GatedCachedNonlinearConnection]";
	}

private:
	// performance acceleration ONLY
	vector<float> cachedOutvals;
};


/**
 * outLayer = gateLayer * tanh(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * tanh(inLayer[t - temporalSkip])
 */
typedef GatedCachedNonlinearConnection<lmn::tanh, lmn::tanhGradient>
	GatedTanhConnection;


/**
 * outLayer = gateLayer * sigmoid(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * sigmoid(inLayer[t - temporalSkip])
 */
typedef GatedCachedNonlinearConnection<lmn::sigmoid, lmn::sigmoidGradient>
	GatedSigmoidConnection;

#endif /* GATED_CONNECTION_H_ */
