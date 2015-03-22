/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "global_utils.h"
#include "layer.h"

class Connection
{
public:
	Connection(Layer& _inLayer, Layer& _outLayer):
		inLayer(_inLayer), outLayer(_outLayer)
    {
    }

	virtual ~Connection() {}

	void forward()
	{
		_forward(inLayer.outValue, outLayer.inValue);
	}

	void backward()
	{
		_backward(inLayer.outValue, inLayer.outGradient, outLayer.inGradient);
	}

	virtual void _forward(float& inlayerOutval, float& outlayerInval) = 0;

	virtual void _backward(float& inlayerOutval, float& inlayerOutgrad, float& outlayerIngrad) = 0;

protected:
	Layer& inLayer;
	Layer& outLayer;
};

class ConstantConnection : public Connection
{
public:
	ConstantConnection(Layer& _inLayer, Layer& _outLayer):
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
};

class LinearConnection : public Connection
{
public:
	LinearConnection(Layer& _inLayer, Layer& _outLayer):
		Connection(_inLayer, _outLayer)
	{
		// TODO random number
		param = 3.73;
	}

	~LinearConnection() {}

	void _forward(float& inlayerOutval, float& outlayerInval)
	{

	}

	void _backward(float& inlayerOutval, float& inlayerOutgrad, float& outlayerIngrad)
	{

	}

	float param;
};

#endif /* CONNECTION_H_ */
