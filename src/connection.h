/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "layer.h"
#include "component.h"
#include "parameter.h"
#include "engine/tensor.h"
#include "utils/global_utils.h"
#include "utils/rand_utils.h"

/**
 * Contains the actual parameters
 */
class Connection : public Component
{
public:
	Connection(Layer::Ptr inLayer, Layer::Ptr outLayer):
		inLayer(inLayer), outLayer(outLayer)
    {
    }

	virtual ~Connection() {};

	virtual void forward(int inFrame = 0, int outFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);
		this->inFrame_ = inFrame;
		this->outFrame_ = outFrame;

		forward_impl(inLayer->out_value(inFrame), outLayer->in_value(outFrame));
	}

	/**
	 * handle h[-1], h[-2] ... h[-maxTemporalSkip]
	 */
	virtual void prehistory_forward(ParamContainer::Ptr pcontainer, int inFrame, int outFrame)
	{
		LMN_ASSERT_THROW(inFrame < 0,
				NetworkException("inFrame should be < 0 for prehistory_forward"));

		this->inFrame_ = inFrame;
		this->outFrame_ = outFrame;

		// inFrame < 0, python-style indexing from the right end
		inFrame += pcontainer->size();
		forward_impl(*pcontainer->param_value_ptr(inFrame),
			outLayer->in_value(outFrame));
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);
		this->inFrame_ = inFrame;
		this->outFrame_ = outFrame;

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();

		backward_impl(outLayer->in_gradient(
					isHistorySaved ? outFrame : 0),
				inLayer->out_value(inFrame),
				inLayer->out_gradient(
					isHistorySaved ? inFrame : outFrame - inFrame));
	}

	/**
	 * Back prop to h[-1] ...
	 */
	virtual void prehistory_backward(ParamContainer::Ptr pcontainer, int outFrame, int inFrame)
	{
		LMN_ASSERT_THROW(inFrame < 0,
				NetworkException("inFrame should be < 0 for prehistory_backward"));
		this->inFrame_ = inFrame;
		this->outFrame_ = outFrame;

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();

		// inFrame < 0, python-style indexing from the right end
		inFrame += pcontainer->size();
		backward_impl(outLayer->in_gradient(
					isHistorySaved ? outFrame : 0),
				*pcontainer->param_value_ptr(inFrame),
				*pcontainer->param_gradient_ptr(inFrame));
	}

	virtual void forward_impl(
			Tensor& inlayerOutval, Tensor& outlayerInval) = 0;

	virtual void backward_impl(
			Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& inlayerOutgrad) = 0;

	virtual void zero_clear()
	{}

	/**
	 * Read only. Forward/backward latest frame number
	 */
	int in_frame() { return this->inFrame_; }
	int out_frame() { return this->outFrame_; }

	/************************************/
	TYPEDEF_PTR(Connection);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(Connection)

	GEN_DOWN_CAST_STATIC_MEMBER(Connection)

	Layer::Ptr inLayer;
	Layer::Ptr outLayer;

protected:
	/**
	 * TODO anything to init for connections?
	 * Implements Component::initialize
	 */
	virtual void initialize_impl() { }

	// Helper for backward/forward in/outLayer check
	void check_frame_consistency(int inFrame, int outFrame)
	{
		LMN_ASSERT_THROW(inFrame >= 0 && outFrame >= 0,
			NetworkException("Both inFrame and outFrame must be positive.\n"
							"Otherwise use prehistory_forward"));

		LMN_ASSERT_THROW(
			inLayer->max_temporal_skip() == outLayer->max_temporal_skip(),
			NetworkException("inLayer must have the same maxTemporalSkip as outLayer"));

		if (!inLayer->is_full_gradient_history_saved())
		{
			LMN_ASSERT_THROW(
				inFrame <= outFrame && outFrame <= inFrame + inLayer->max_temporal_skip(),
				NetworkException("Inconsistency: inFrame <= outFrame <= inFrame + layer.maxTemporalSkip"));
		}
	}

private:
	int inFrame_ = 0, outFrame_ = 0;
};

TYPEDEF_PTR_EXTERNAL(Connection);

#endif /* CONNECTION_H_ */
