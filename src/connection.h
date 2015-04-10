/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "global_utils.h"
#include "rand_utils.h"
#include "math_utils.h"
#include "layer.h"
#include "component.h"
#include "parameter.h"

/**
 * Contains the actual parameters
 */
class Connection : public Component
{
public:
	Connection(Layer::Ptr _inLayer, Layer::Ptr _outLayer):
		inLayer(_inLayer), outLayer(_outLayer)
    {
    }

	virtual ~Connection() =default;

	virtual void forward(int inFrame = 0, int outFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		forward_impl(inLayer->outValues[inFrame], outLayer->inValues[outFrame]);
	}

	/**
	 * handle h[-1], h[-2] ... h[-maxTemporalSkip]
	 */
	virtual void prehistory_forward(ParamContainer::Ptr pcontainer, int inFrame, int outFrame)
	{
		assert_throw(inFrame < 0,
			NetworkException("inFrame should be < 0 for prehistory_forward"));
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		forward_impl(vec_at(pcontainer->paramValues, inFrame),
			outLayer->inValues[outFrame]);
	}

	virtual void backward(int outFrame = 0, int inFrame = 0)
	{
		check_frame_consistency(inFrame, outFrame);
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();

		backward_impl(outLayer->inGradients[
					isHistorySaved ? outFrame : 0],
				inLayer->outValues[inFrame],
				inLayer->outGradients[
					isHistorySaved ? inFrame : outFrame - inFrame]);
	}

	/**
	 * Back prop to h[-1] ...
	 */
	virtual void prehistory_backward(ParamContainer::Ptr pcontainer, int outFrame, int inFrame)
	{
		assert_throw(inFrame < 0,
			NetworkException("inFrame should be < 0 for prehistory_backward"));
		this->_inFrame = inFrame;
		this->_outFrame = outFrame;

		bool isHistorySaved = inLayer->is_full_gradient_history_saved();

		backward_impl(outLayer->inGradients[
					isHistorySaved ? outFrame : 0],
				vec_at(pcontainer->paramValues, inFrame),
				vec_at(pcontainer->paramGradients, inFrame));
	}

	virtual void forward_impl(float inlayerOutval, float& outlayerInval) = 0;

	virtual void backward_impl(float& outlayerIngrad, float& inlayerOutval, float& inlayerOutgrad) = 0;

	virtual void reset() {}

	/**
	 * Read only. Forward/backward latest frame number
	 */
	int in_frame() { return this->_inFrame; }
	int out_frame() { return this->_outFrame; }

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

	Layer::Ptr inLayer;
	Layer::Ptr outLayer;
protected:

	// Helper for backward/forward in/outLayer check
	void check_frame_consistency(int inFrame, int outFrame)
	{
		assert_throw(inFrame >= 0 && outFrame >= 0,
			NetworkException("Both inFrame and outFrame must be positive.\n"
				"Otherwise use prehistory_forward"));

		assert_throw(
			inLayer->get_max_temporal_skip() == outLayer->get_max_temporal_skip(),
			NetworkException(
				"inLayer must have the same maxTemporalSkip as outLayer"));

		if (!inLayer->is_full_gradient_history_saved())
		{
			assert_throw(
				inFrame <= outFrame && outFrame <= inFrame + inLayer->get_max_temporal_skip(),
				NetworkException(
					"Inconsistency: inFrame <= outFrame <= inFrame + layer.maxTemporalSkip"));
		}
	}

private:
	int _inFrame = 0, _outFrame = 0;
};

TypedefPtr(Connection);

#endif /* CONNECTION_H_ */
