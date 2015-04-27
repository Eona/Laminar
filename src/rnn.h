/*
 * Eona Studio (c) 2015
 */

#ifndef RNN_H_
#define RNN_H_

#include "network.h"

/**************************************
******* Recurrent Network *********
**************************************/
/**
 * WARNING NOTE topology limitation:
 * A layer cannot:
 * (1) End in itself without any outgoing connection to loss layer
 * (2) Only have recurrent connections to loss layer, no forward connection
 *
 * Example of a bad recurrent topology:
 *
 	auto linput = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<ScalarLayer>(TARGET_DIM, 1.3f);
	auto l3 = Layer::make<CosineLayer>(TARGET_DIM); // gate
	auto lloss = Layer::make<SquareLossLayer>(TARGET_DIM);

	net.add_layer(linput);
	net.new_connection<FullConnection>(linput, l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(linput, l3);
	net.add_layer(l3);
	// l2 is connected to lloss only by a recurrent connection: not allowed!
	net.add_recurrent_connection<GatedTanhConnection>(l2, l3, lloss);
	net.add_layer(lloss);
 *
 * In the above example, l2 is never connected to lloss in the current time frame,
 * it's an end in itself. (The recurrent connection is from the last frame)
 * Backprop will fail because l2's out_gradient() is never set.
 */
class RecurrentNetwork : public Network
{
public:
	RecurrentNetwork(EngineBase::Ptr engine, DataManagerBase::Ptr dataManager,
			int historyLength, int maxTemporalSkip = 1) :
		Network(engine, dataManager),
		historyLength(historyLength),
		maxTemporalSkip(maxTemporalSkip)
	{ }

	virtual ~RecurrentNetwork() {};

	virtual void init_max_temporal_skip(int maxTemporalSkip)
	{
		initGuard.assert_before_initialize("init_max_temporal_skip", "RecurrentNetwork");
		this->maxTemporalSkip = maxTemporalSkip;
	}

	int max_temporal_skip()
	{
		return this->maxTemporalSkip;
	}

	virtual void init_history_length(int historyLength)
	{
		initGuard.assert_before_initialize("init_history_length", "RecurrentNetwork");
		this->historyLength = historyLength;
	}

	int history_length()
	{
		return this->historyLength;
	}

	virtual void add_recur_connection(Connection::Ptr conn, int temporalSkip = 1)
	{
		LMN_ASSERT_THROW(maxTemporalSkip == Layer::UNLIMITED_TEMPORAL_SKIP
						|| temporalSkip <= maxTemporalSkip,
			NetworkException("temporalSkip should be <= maxTemporalSkip.\n"
				"Use init_max_temporal_skip() to change the upper limit."));

		components.push_back(Component::upcast(conn));
		connections.push_back(conn);
		this->check_add_param_container(conn);

		auto connLayer = conn->inLayer;

		// Always store the largest temporalSkip seen so far associated with the layer
		// prehistoryMapHelper will be used to initialize prehistoryMap, which contains
		// ParamContainer for learnable prehistory params.
		if (!key_exists(prehistoryMapHelper, connLayer)
			|| prehistoryMapHelper[connLayer] < temporalSkip)
			prehistoryMapHelper[connLayer] = temporalSkip;

		recurConnTemporalSkipMap[conn] = temporalSkip;
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_recur_connection(ArgT&& ... args)
	{
		this->add_recur_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...));
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_recur_skip_connection(int temporalSkip, ArgT&& ... args)
	{
		this->add_recur_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...),
			temporalSkip);
	}

	/*********** Ptr ***********/
	TYPEDEF_PTR(RecurrentNetwork);

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(RecurrentNetwork)

protected:
	/**************************************
	******* Training logic *********
	**************************************/
	virtual void load_input()
	{
		for (int frame = 0; frame < this->historyLength; ++frame)
		{
			dataManager->upload_input(layers[0]->in_value(frame));
			// TODO has to upload an instruction because the compiled executable
			// cannot run a non-registered instruction
			if (frame != this->historyLength - 1)
				dataManager->upload_prepare_next_batch();
		}

	}

	virtual void load_target()
	{
		for (int frame = 0; frame < this->historyLength; ++frame)
		{
			dataManager->upload_target(lossLayer->target_value(frame));
			if (frame != this->historyLength - 1)
				dataManager->upload_prepare_next_batch();
		}
	}

	virtual void zero_clear()
	{
		for (Component::Ptr compon : this->components)
			compon->zero_clear();

		for (auto& entry : prehistoryMap)
			entry.second->clear_gradients();

//		frame = 0;
	}

	/**
	 * First feeds forward in current time frame,
	 * then props to the next time frame
	 */
	virtual void forward()
	{
		for (int frame = 0; frame < this->historyLength; ++ frame)
		{
			for (Component::Ptr compon : this->components)
			{
				Connection::Ptr conn = Component::cast<Connection>(compon);
				if (conn && key_exists(recurConnTemporalSkipMap, conn))
				{
					int skip = recurConnTemporalSkipMap[conn];
					if (frame >= skip)
						conn->forward(frame - skip, frame);
					else
						conn->prehistory_forward(
							// Ugly workaround for eclipse syntax highlighter
							std::static_pointer_cast<ParamContainer>(prehistoryMap[conn->inLayer]),
							frame - skip, frame);
				}
				else
					compon->forward(frame, frame);
			}
		}
	}

	/**
	 * First back-props to the previous time point,
	 * then pass the gradient backward in current time.
	 */
	virtual void backward()
	{
		for (int frame = this->historyLength - 1; frame >= 0; --frame)
		{
			for (int i = components.size() - 1; i >= 0; --i)
			{
				Component::Ptr compon = components[i];
				Connection::Ptr conn = Component::cast<Connection>(compon);
				if (key_exists(recurConnTemporalSkipMap, conn))
				{
					int skip = recurConnTemporalSkipMap[conn];
					if (frame >= skip)
						conn->backward(frame, frame - skip);
					else
						conn->prehistory_backward(
							std::static_pointer_cast<ParamContainer>(prehistoryMap[conn->inLayer]),
							frame, frame - skip);
				}
				else
					compon->backward(frame, frame);
			}

			for (Layer::Ptr layer : layers)
				layer->shift_back_gradient_window();
		}
	}

	virtual void initialize()
	{
		// initialize prehistory ParamContainers
		this->init_prehistory_params();

		for (Layer::Ptr l : this->layers)
			l->init_history_length(this->historyLength);

		for (Layer::Ptr layer : layers)
			layer->init_max_temporal_skip(this->maxTemporalSkip);

		Network::initialize();
	}

	/**
	 * Initialize prehistory ParamContainers and fill with random init values
	 * Use prehistoryMapHelper to initialize prehistoryMap
	 */
	void init_prehistory_params()
	{
		for (auto helperEntry : this->prehistoryMapHelper)
		{
			auto layer = helperEntry.first;
			int temporalSkip = helperEntry.second;

			auto h_0 = ParamContainer::make(temporalSkip);

			for (int paramIdx = 0; paramIdx < h_0->size(); ++paramIdx)
			{
				// Construct actual tensors for ParamContainer
				auto& valuePtr = h_0->param_value_ptr(paramIdx);
				auto& gradPtr = h_0->param_gradient_ptr(paramIdx);

				// prehistory dimension augmented by minibatch size
				Dimension dimWithBatch = vec_augment(layer->dim(), dataManager->batch_size());

				valuePtr = Tensor::make(engine, dimWithBatch);
				gradPtr = Tensor::make(engine, dimWithBatch);

				// Randomly initialize prehistory
				lmn::fill_rand_prehistory(*valuePtr);
			}

			this->prehistoryMap[layer] = h_0;
			this->paramContainers.push_back(h_0);
		}
	}

protected:
//	int frame = 0;
	int historyLength;
	int maxTemporalSkip = 1;

	/**
	 * Map layer to its prehistory parameters
	 */
	std::unordered_map<Layer::Ptr, ParamContainer::Ptr> prehistoryMap;

	/**
	 * Map layer to its largest temporal skip value.
	 * Helps initialize prehistoryMap
	 */
	std::unordered_map<Layer::Ptr, int> prehistoryMapHelper;

	/**
	 * Map recurrent connection to its temporal skip value
	 */
	std::unordered_map<Connection::Ptr, int> recurConnTemporalSkipMap;
};


template<typename T>
typename std::enable_if<std::is_base_of<RecurrentNetwork, T>::value, std::ostream>::type&
operator<<(std::ostream& os, T& net)
{
	os << "[RecurrentNet\n";
	for (auto compon : net.components)
		os << "  " << string(*compon) << "\n";
	os << " " << "recurrent connections:\n";
//	for (auto connInfo : net.recurConnectionInfos)
//		os << "  " << connInfo.conn->str() << "\n";
//	os << "]";
	return os;
}

#endif /* RNN_H_ */
