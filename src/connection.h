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
		resize_on_demand(inLayer->outValues, inFrame);
		resize_on_demand(outLayer->inValues, outFrame);

		_forward(inLayer->outValues[inFrame], outLayer->inValues[outFrame]);
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		resize_on_demand(outLayer->inGradients, outFrame);
		resize_on_demand(inLayer->outGradients, inFrame);

		_backward(outLayer->inGradients[outFrame], inLayer->outValues[inFrame], inLayer->outGradients[inFrame]);
	}

	virtual void _forward(float& inlayerOutval, float& outlayerInval) = 0;

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

protected:
	LayerPtr inLayer;
	LayerPtr outLayer;
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

	void _forward(float& inlayerOutval, float& outlayerInval)
	{
		outlayerInval = inlayerOutval;
	}

	void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad)
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
		gradient(paramGradients[0]),
		rnd(-3, 6)
	{
		param = rnd();
		param = fakernd();
	}

	~LinearConnection() {}

	void _forward(float& inlayerOutval, float& outlayerInval)
	{
		// NOTE matrix multiplication order applies here
		outlayerInval += param * inlayerOutval;
	}

	void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad)
	{
		// NOTE matrix multiplication order applies here
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

	UniformRand<float> rnd;
	// NOTE debug only
	FakeRand& fakernd = FakeRand::instance();

	float& param; // aliases
	float& gradient;
};

ostream& operator<<(ostream& os, LinearConnection& conn)
{
	os << conn.str();
	return os;
}
ostream& operator<<(ostream& os, LinearConnection&& conn)
{
	os << conn.str();
	return os;
}

ostream& operator<<(ostream& os, ConstantConnection& conn)
{
	os << conn.str();
	return os;
}

ostream& operator<<(ostream& os, ConstantConnection&& conn)
{
	os << conn.str();
	return os;
}

#endif /* CONNECTION_H_ */
