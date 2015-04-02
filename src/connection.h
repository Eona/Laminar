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

	virtual void forward(int inTime, int outTime)
	{
		_forward(inLayer->outValue[inTime], outLayer->inValue[outTime]);
	}

	virtual void forward()
	{
		this->forward(0, 0);
	}

	virtual void backward(int inTime, int outTime)
	{
		_backward(inLayer->outValue[inTime], inLayer->outGradient[inTime], outLayer->inGradient[outTime]);
	}

	virtual void backward()
	{
		this->backward(0, 0);
	}

	virtual void _forward(float& inlayerOutval, float& outlayerInval) = 0;

	virtual void _backward(float& inlayerOutval, float& inlayerOutgrad, float& outlayerIngrad) = 0;

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

	void _backward(float& inlayerOutval, float& inlayerOutgrad, float& outlayerIngrad)
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
		gradient(0.0f),
		rand(-3, 6)
	{
		param = rand();
	}

	~LinearConnection() {}

	void _forward(float& inlayerOutval, float& outlayerInval)
	{
		// NOTE matrix multiplication order applies here
		outlayerInval += param * inlayerOutval;
	}

	void _backward(float& inlayerOutval, float& inlayerOutgrad, float& outlayerIngrad)
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
	UniformRand<float> rand;
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
