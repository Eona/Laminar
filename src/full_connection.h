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
		param(ParamContainer::get_param_value(0)),
		gradient(ParamContainer::get_param_gradient(0))
	{
		assert_throw(inLayer->dim().size() == 1
				&& outLayer->dim().size() == 1,
			ComponentException("FullConnection requires the in/outLayers to be one-dimensional"));
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

	virtual void zero_clear()
	{
		ParamContainer::clear_gradients();
	}

	Tensor::Ptr& param; // aliases
	Tensor::Ptr& gradient;

protected:
	virtual void initialize_impl()
	{
		auto dims = { outLayer->dim()[0], inLayer->dim()[0] };
		param = Tensor::make(engine, dims);
		gradient = Tensor::make(engine, dims);
		lmn::fill_rand(*param);
	}
};

TYPEDEF_PTR_EXTERNAL(FullConnection);

#endif /* FULL_CONNECTION_H_ */
