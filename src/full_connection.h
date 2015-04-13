/*
 * Eona Studio (c) 2015
 */

#ifndef FULL_CONNECTION_H_
#define FULL_CONNECTION_H_

#include "connection.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

class ConstantConnection : public Connection
{
public:
	ConstantConnection(Layer::Ptr _inLayer, Layer::Ptr _outLayer):
		Connection(_inLayer, _outLayer)
	{
	}

	virtual ~ConstantConnection() {};

	virtual void forward_impl(Tensor& inlayerOutval, Tensor& outlayerInval)
	{
		outlayerInval = inlayerOutval;
	}

	virtual void backward_impl(Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& inlayerOutgrad)
	{
		inlayerOutgrad += outlayerIngrad;
	}

	virtual explicit operator string() const
	{
		return "[ConstantConnection]";
	}
};

class FullConnection : public Connection, public ParamContainer
{
public:
	FullConnection(Layer::Ptr _inLayer, Layer::Ptr _outLayer):
		Connection(_inLayer, _outLayer),
		ParamContainer(1),
		param(paramValues[0]),
		gradient(paramGradients[0])
	{
//		param = fakernd();
		assert_throw(inLayer->dim().size() == 1
				&& outLayer->dim().size() == 1,
			ComponentException("FullConnection requires the in/outLayers to be one-dimensional"));

		auto dims = { outLayer->dim()[0], inLayer->dim()[0] };
		param = Tensor::make(dims);
		gradient = Tensor::make(dims);

		lmn::fill_rand(*param);
	}

	virtual ~FullConnection() {};

	virtual void forward_impl(Tensor& inlayerOutval, Tensor& outlayerInval)
	{
		outlayerInval += *param * inlayerOutval;
	}

	virtual void backward_impl(Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& inlayerOutgrad)
	{
		// should check if input module actually has gradient
		inlayerOutgrad += lmn::transpose(*param) * outlayerIngrad;
		*gradient += outlayerIngrad * lmn::transpose(inlayerOutval);
	}

	virtual explicit operator string() const
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
		// TODO
//		ParamContainer::reset_gradients();
	}

	// DUMMY
	FakeRand& fakernd = FakeRand::instance_connection();

	Tensor::Ptr& param; // aliases
	Tensor::Ptr& gradient;
};

TYPEDEF_PTR(FullConnection);

#endif /* FULL_CONNECTION_H_ */
