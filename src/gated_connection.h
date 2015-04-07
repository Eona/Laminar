/*
 * Eona Studio (c) 2015
 */


#ifndef GATED_CONNECTION_H_
#define GATED_CONNECTION_H_

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


class GatedTanhConnection : public GatedConnection
{
public:
	/**
	 * outLayer = gateLayer * tanh(inLayer)
	 * If used in a recurrent fashion, inLayer will be from the past while
	 * gateLayer and outLayer will both be in the current timeframe.
	 * outLayer[t] = gateLayer[t] * tanh(inLayer[t - temporalSkip])
	 */
	GatedTanhConnection(LayerPtr _inLayer, LayerPtr _gateLayer, LayerPtr _outLayer):
		GatedConnection(_inLayer, _gateLayer, _outLayer)
	{ }

	virtual void _gated_forward(float& inlayerOutval, float& gateOutval,
		// output param:
		float& outlayerInval)
	{
		int t = out_frame();
		vec_resize_on_demand(cachedOutvals, t);

		cachedOutvals[t] = lmn::tanh(inlayerOutval);

		outlayerInval += gateOutval * cachedOutvals[t];
	}

	virtual void _gated_backward(float& outlayerIngrad, float& inlayerOutval, float& gateOutval,
			// write to output params:
			float& inlayerOutgrad, float& gateOutgrad)
	{
		float cachedOutval = cachedOutvals[out_frame()];

		inlayerOutgrad += gateOutval * lmn::tanhGradient(cachedOutval) * outlayerIngrad;
		gateOutgrad += outlayerIngrad * cachedOutval;
	}

	string str()
	{
		return "[GatedTanhConnection]";
	}

private:
	// performance acceleration ONLY
	vector<float> cachedOutvals;
};

#endif /* GATED_CONNECTION_H_ */