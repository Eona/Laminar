/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "global_utils.h"
#include "rand_utils.h"
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

		resize_on_demand(inLayer->outValues, inFrame);
		resize_on_demand(outLayer->inValues, outFrame);

		float inlayerOutvalue = 0;
		if (inFrame >= 0)
			inlayerOutvalue = inLayer->outValues[inFrame];

		_forward(inlayerOutvalue, outLayer->inValues[outFrame]);
	}

	/**
	 * handle h[-1], h[-2] ... h[-maxTemporalSkip]
	 */
	virtual void prehistory_forward(ParamContainer::Ptr pcontainer, int inFrame, int outFrame)
	{
		assert_throw(inFrame < 0,
			NetworkException("inFrame should be < 0 for prehistory_forward"));

		resize_on_demand(outLayer->inValues, outFrame);

		_forward(vec_at(pcontainer->paramValues, inFrame),
			outLayer->inValues[outFrame]);
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();
		if (isHistorySaved)
		{
			resize_on_demand(inLayer->outGradients, inFrame);
			resize_on_demand(outLayer->inGradients, outFrame);
		}

		if (inFrame >= 0)
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

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();
		if (isHistorySaved)
			resize_on_demand(outLayer->inGradients, outFrame);

		_backward(outLayer->inGradients[
					isHistorySaved ? outFrame : 0],
				vec_at(pcontainer->paramValues, inFrame),
				vec_at(pcontainer->paramGradients, inFrame));
	}

	virtual void _forward(float inlayerOutval, float& outlayerInval) = 0;

	virtual void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad) = 0;

	virtual void reset() {}

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
		return "[ConstantConn]";
	}
};

class LinearConnection : public Connection, public ParamContainer
{
public:
	LinearConnection(LayerPtr _inLayer, LayerPtr _outLayer):
		Connection(_inLayer, _outLayer),
		ParamContainer(1),
		param(paramValues[0]),
		gradient(paramGradients[0])
	{
		// DUMMY
		param = debugrnd();
		param = fakernd();
	}

	~LinearConnection() {}

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
		os << "[LinearConn: "
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
	UniformFloatSingleton<-1, 2>& debugrnd = UniformFloatSingleton<-1, 2>::instance();
	FakeRand& fakernd = FakeRand::instance();

	float& param; // aliases
	float& gradient;
};

#endif /* CONNECTION_H_ */
