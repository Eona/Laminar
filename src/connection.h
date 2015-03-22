/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "global_utils.h"
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

	virtual void forward()
	{
		_forward(inLayer->outValue, outLayer->inValue);
	}

	virtual void backward()
	{
		_backward(inLayer->outValue, inLayer->outGradient, outLayer->inGradient);
	}

	virtual void _forward(float& inlayerOutval, float& outlayerInval) = 0;

	virtual void _backward(float& inlayerOutval, float& inlayerOutgrad, float& outlayerIngrad) = 0;

protected:
	LayerPtr inLayer;
	LayerPtr outLayer;
};

typedef shared_ptr<Connection> ConnectionPtr;

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
		Connection(_inLayer, _outLayer), gradient(0.0f)
	{
		// TODO random number
		param = 3.73;
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

	float param;
	float gradient;
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
