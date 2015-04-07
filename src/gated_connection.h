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
	GatedConnection(LayerPtr _inLayer, LayerPtr _gateLayer, LayerPtr _outLayer):
		Connection(_inLayer, _outLayer),
		gateLayer(_gateLayer)
	{ }

	virtual void _forward(float inlayerOutval, float& outlayerInval)
	{
		_gated_forward(inlayerOutval, gateLayer->outValues[out_frame()],
				// output param:
				outlayerInval);
	}

	virtual void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad)
	{
		_gated_backward(outlayerIngrad, inlayerOutval, gateLayer->outValues[out_frame()],
				// output params:
				inlayerOutgrad,
				gateLayer->outGradients[
						gateLayer->is_full_gradient_history_saved() ?
								out_frame() : 0]);
	}

	/*********** Subclasses should override following ***********/
	virtual void _gated_forward(float& inlayerOutval, float& gateOutval,
			// output param:
			float& outlayerInval)
	{
		outlayerInval += gateOutval * inlayerOutval;
	}

	virtual void _gated_backward(float& outlayerIngrad, float& inlayerOutval, float& gateOutval,
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
	LayerPtr gateLayer;
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
class GatedCachedNonlinearConnection : public GatedConnection
{
public:
	GatedCachedNonlinearConnection(
			LayerPtr _inLayer, LayerPtr _gateLayer, LayerPtr _outLayer,
			lmn::TransferFunction _nonlinear,
			lmn::TransferFunction _nonlinearGradient):
		GatedConnection(_inLayer, _gateLayer, _outLayer),
		nonlinear(_nonlinear),
		nonlinearGradient(_nonlinearGradient)
	{ }

	virtual void _gated_forward(float& inlayerOutval, float& gateOutval,
		// output param:
		float& outlayerInval)
	{
		int t = out_frame();
		vec_resize_on_demand(cachedOutvals, t);

		cachedOutvals[t] = this->nonlinear(inlayerOutval);

		outlayerInval += gateOutval * cachedOutvals[t];
	}

	virtual void _gated_backward(float& outlayerIngrad, float& inlayerOutval, float& gateOutval,
			// write to output params:
			float& inlayerOutgrad, float& gateOutgrad)
	{
		float cachedOutval = cachedOutvals[out_frame()];

		inlayerOutgrad += gateOutval * this->nonlinearGradient(cachedOutval) * outlayerIngrad;
		gateOutgrad += outlayerIngrad * cachedOutval;
	}

	string str()
	{
		return "[GatedCachedNonlinearConnection]";
	}

protected:
	lmn::TransferFunction nonlinear;
	lmn::TransferFunction nonlinearGradient;

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
class GatedTanhConnection : public GatedCachedNonlinearConnection
{
public:
	GatedTanhConnection(
			LayerPtr _inLayer, LayerPtr _gateLayer, LayerPtr _outLayer):
		GatedCachedNonlinearConnection(_inLayer, _gateLayer, _outLayer,
				lmn::tanh,
				lmn::tanhGradient)
	{ }

	string str()
	{
		return "[GatedTanhConnection]";
	}
};

/**
 * outLayer = gateLayer * sigmoid(inLayer)
 * If used in a recurrent fashion, inLayer will be from the past while
 * gateLayer and outLayer will both be in the current timeframe.
 * outLayer[t] = gateLayer[t] * sigmoid(inLayer[t - temporalSkip])
 */
class GatedSigmoidConnection : public GatedCachedNonlinearConnection
{
public:
	GatedSigmoidConnection(
			LayerPtr _inLayer, LayerPtr _gateLayer, LayerPtr _outLayer):
		GatedCachedNonlinearConnection(_inLayer, _gateLayer, _outLayer,
				lmn::sigmoid,
				lmn::sigmoidGradient)
	{ }

	string str()
	{
		return "[GatedSigmoidConnection]";
	}
};

#endif /* GATED_CONNECTION_H_ */
