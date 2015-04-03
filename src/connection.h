/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "global_utils.h"
#include "rand_utils.h"
#include "layer.h"
#include "component.h"

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

	virtual void forward(int inTime = 0, int outTime = 0)
	{
		resize_on_demand(inLayer->outValue, inTime);
		resize_on_demand(outLayer->inValue, outTime);

		_forward(inLayer->outValue[inTime], outLayer->inValue[outTime]);
	}

	virtual void backward(int outTime = 0, int inTime = 0)
	{
		resize_on_demand(outLayer->inGradient, outTime);
		resize_on_demand(inLayer->outGradient, inTime);

		_backward(outLayer->inGradient[outTime], inLayer->outValue[inTime], inLayer->outGradient[inTime]);
	}

	virtual void _forward(float& inlayerOutval, float& outlayerInval) = 0;

	virtual void _backward(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad) = 0;

	virtual void reset() {}

protected:
	LayerPtr inLayer;
	LayerPtr outLayer;
};

TypedefPtr(Connection);

/**
 * Make a polymorphic shared pointer
 */
template<typename ConnectionT, typename ...ArgT>
ConnectionPtr make_connection(ArgT&& ... args)
{
	return static_cast<ConnectionPtr>(
			std::make_shared<ConnectionT>(
					std::forward<ArgT>(args) ...));
}

/**
 * Down cast ConnectionPtr to a specific connection type
 */
template<typename ConnectionT>
shared_ptr<ConnectionT> cast_connection(ConnectionPtr conn)
{
	return std::dynamic_pointer_cast<ConnectionT>(conn);
}

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

class LinearConnection : public Connection
{
public:
	LinearConnection(LayerPtr _inLayer, LayerPtr _outLayer):
		Connection(_inLayer, _outLayer),
		gradient(0.0f)
//		rnd(-3, 6) NOTE
	{
		param = rnd();
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
		inlayerOutgrad += transpose(param) * outlayerIngrad;
		this->gradient += outlayerIngrad * transpose(inlayerOutval);
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
		gradient = 0;
	}

	float param;
	float gradient;
	// NOTE debug only
//	UniformRand<float> rnd;
	FakeRand<float>& rnd = FakeRand<float>::instance();
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
