/*
 * Eona Studio (c) 2015
 */

#ifndef FULL_CONNECTION_H_
#define FULL_CONNECTION_H_

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

TypedefPtr(FullConnection);

#endif /* FULL_CONNECTION_H_ */
