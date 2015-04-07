/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "global_utils.h"
#include "rand_utils.h"
#include "math_utils.h"
#include "layer.h"
#include "component.h"
#include "parameter.h"

/**
 * Contains the actual parameters
 */
class Connection : public Component
{
public:
	Connection(LayerPtr _inLayer, LayerPtr _outLayer):
		inLayer(_inLayer), outLayer(_outLayer)
    {
    }

	virtual ~Connection() {}

	virtual void forward(int inFrame = 0, int outFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		_forward(inLayer->outValues[inFrame], outLayer->inValues[outFrame]);
	}

	/**
	 * handle h[-1], h[-2] ... h[-maxTemporalSkip]
	 */
	virtual void prehistory_forward(ParamContainer::Ptr pcontainer, int inFrame, int outFrame)
	{
		assert_throw(inFrame < 0,
			NetworkException("inFrame should be < 0 for prehistory_forward"));
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		_forward(vec_at(pcontainer->paramValues, inFrame),
			outLayer->inValues[outFrame]);
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();

		_backward(outLayer->inGradients[
					isHistorySaved ? outFrame : 0],
				inLayer->outValues[inFrame],
				inLayer->outGradients[
					isHistorySaved ? inFrame : outFrame - inFrame]);
	}

	/**
	 * Back prop to h[-1] ...
	 */
	virtual void prehistory_backward(ParamContainer::Ptr pcontainer, int outFrame, int inFrame)
	{
		assert_throw(inFrame < 0,
			NetworkException("inFrame should be < 0 for prehistory_backward"));
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();

		_backward(outLayer->inGradients[
					isHistorySaved ? outFrame : 0],
				vec_at(pcontainer->paramValues, inFrame),
				vec_at(pcontainer->paramGradients, inFrame));
	}

	virtual void _forward(float inlayerOutval, float& outlayerInval) = 0;

	virtual void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad) = 0;

	virtual void reset() {}

	/**
	 * Read only. Forward/backward latest frame number
	 */
	int in_frame() { return this->_inFrame; }
	int out_frame() { return this->_outFrame; }

	/************************************/
	typedef shared_ptr<Connection> Ptr;

	/**
	 * Make a polymorphic shared pointer
	 */
	template<typename ConnectionT, typename ...ArgT>
	static Connection::Ptr make(ArgT&& ... args)
	{
		return static_cast<Connection::Ptr>(
				std::make_shared<ConnectionT>(
						std::forward<ArgT>(args) ...));
	}

	/**
	 * Down cast ConnectionPtr to a specific connection type
	 */
	template<typename ConnectionT>
	static shared_ptr<ConnectionT> cast(Connection::Ptr conn)
	{
		return std::dynamic_pointer_cast<ConnectionT>(conn);
	}

	LayerPtr inLayer;
	LayerPtr outLayer;
protected:

	// Helper for backward/forward in/outLayer check
	void check_frame_consistency(int inFrame, int outFrame)
	{
		assert_throw(inFrame >= 0 && outFrame >= 0,
			NetworkException("Both inFrame and outFrame must be positive.\n"
				"Otherwise use prehistory_forward"));

		assert_throw(
			inLayer->get_max_temporal_skip() == outLayer->get_max_temporal_skip(),
			NetworkException(
				"inLayer must have the same maxTemporalSkip as outLayer"));

		if (!inLayer->is_full_gradient_history_saved())
		{
			assert_throw(
				inFrame <= outFrame && outFrame <= inFrame + inLayer->get_max_temporal_skip(),
				NetworkException(
					"Inconsistency: inFrame <= outFrame <= inFrame + layer.maxTemporalSkip"));
		}
	}

private:
	int _inFrame = 0, _outFrame = 0;
};

TypedefPtr(Connection);

class ConstantConnection : public Connection
{
public:
	ConstantConnection(LayerPtr _inLayer, LayerPtr _outLayer):
		Connection(_inLayer, _outLayer)
	{
	}

	~ConstantConnection() {}

	virtual void _forward(float inlayerOutval, float& outlayerInval)
	{
		outlayerInval = inlayerOutval;
	}

	virtual void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad)
	{
		inlayerOutgrad = outlayerIngrad;
	}

	string str()
	{
		return "[ConstantConnection]";
	}
};

class FullConnection : public Connection, public ParamContainer
{
public:
	FullConnection(LayerPtr _inLayer, LayerPtr _outLayer):
		Connection(_inLayer, _outLayer),
		ParamContainer(1),
		param(paramValues[0]),
		gradient(paramGradients[0])
	{
		param = fakernd();
	}

	~FullConnection() {}

	virtual void _forward(float inlayerOutval, float& outlayerInval)
	{
		outlayerInval += param * inlayerOutval;
	}

	virtual void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad)
	{
		// should check if input module actually has gradient
		inlayerOutgrad += lmn::transpose(param) * outlayerIngrad;
		this->gradient += outlayerIngrad * lmn::transpose(inlayerOutval);
	}

	string str()
	{
		ostringstream os;
		os << "[FullConnection: "
			<< "\n\tparam=" << this->param
			<< "\tgrad=" << this->gradient
			<< "]";
		return os.str();
	}

	void reset()
	{
		ParamContainer::resetGradients();
	}

	// DUMMY
	FakeRand& fakernd = FakeRand::instance_connection();

	float& param; // aliases
	float& gradient;
};


class GatedConnection : public Connection
{
public:
	/**
	 * outLayer = inLayer * gateLayer
	 * If used in a recurrent fashion, inLayer will be from the past while
	 * gateLayer and outLayer will both be in the current timeframe.
	 * outLayer[t] = inLayer[t - temporalSkip] * gateLayer[t]
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
	 * outLayer = inLayer * gateLayer
	 * If used in a recurrent fashion, inLayer will be from the past while
	 * gateLayer and outLayer will both be in the current timeframe.
	 * outLayer[t] = inLayer[t - temporalSkip] * gateLayer[t]
	 */
	GatedTanhConnection(LayerPtr _inLayer, LayerPtr _gateLayer, LayerPtr _outLayer):
		GatedConnection(_inLayer, _gateLayer, _outLayer)
	{ }

	virtual void _forward(float inlayerOutval, float& outlayerInval)
	{
		outlayerInval += gateLayer->outValues[out_frame()] * lmn::tanh(inlayerOutval);
	}

	virtual void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad)
	{
		inlayerOutgrad += gateLayer->outValues[out_frame()] * outlayerIngrad;

		bool isHistorySaved = gateLayer->is_full_gradient_history_saved();

		gateLayer->outGradients[
			isHistorySaved ? out_frame() : 0] += outlayerIngrad * inlayerOutval;
	}

	string str()
	{
		return "[GatedTanhConnection]";
	}

	LayerPtr gateLayer;
};

#endif /* CONNECTION_H_ */
