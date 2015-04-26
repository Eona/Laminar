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
	ConstantConnection(Layer::Ptr inLayer, Layer::Ptr outLayer):
		Connection(inLayer, outLayer)
	{
		LMN_ASSERT_THROW(inLayer->dim() == outLayer->dim()
				// we disable the check if either in/outLayer is a DebugLayer
				|| is_castable<DebugLayer>(inLayer)
				|| is_castable<DebugLayer>(outLayer),
			ComponentException("ConstantConnection inLayer and outLayer dim diagrees:\n"
				+ container2str(inLayer->dim()) + " <-> " + container2str(outLayer->dim())));
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
	FullConnection(Layer::Ptr inLayer, Layer::Ptr outLayer):
		Connection(inLayer, outLayer),
		ParamContainer(1),
		param(ParamContainer::param_value_ptr(0)),
		gradient(ParamContainer::param_gradient_ptr(0))
	{
		LMN_ASSERT_THROW(inLayer->dim().size() == 1
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
		std::ostringstream os;
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
