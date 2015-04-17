/*
 * Eona Studio (c) 2015
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "component.h"
#include "composite.h"
#include "layer.h"
#include "connection.h"
#include "engine/engine.h"
#include "engine/tensor.h"
#include "engine/data_manager.h"

class Network
{
public:
	Network(EngineBase::Ptr engine_, DataManagerBase::Ptr dataManager_)
		: engine(engine_), dataManager(dataManager_)
	{
		assert_throw(engine == dataManager->get_engine(),
				NetworkException("DataManager has a different engine"));

		/**
		 * Tag the member methods with their names
		 * These methods only deal with the logic, not computation
		 * All they do is to upload instructions to engine
		 */
		networkMethodMap["initialize"] = &Network::initialize;
		networkMethodMap["forward"] = &Network::forward;
		networkMethodMap["backward"] = &Network::backward;
	}

	virtual ~Network() {};

	virtual void add_layer(Layer::Ptr layer)
	{
		components.push_back(Component::upcast(layer));
		layers.push_back(layer);

		this->check_add_param_container(layer);
	}

	virtual void add_connection(Connection::Ptr conn)
	{
		components.push_back(Component::upcast(conn));
		connections.push_back(conn);

		this->check_add_param_container(conn);
	}

	/**
	 * Cannot add a composite that requires a more specialized network.
	 */
	template<typename CompositeT>
	void add_composite(std::shared_ptr<CompositeT> composite)
	{
		static_assert(is_composite<CompositeT>(),
				"Not a valid composite type");

		composite->manipulate(this);
	}

	template<typename CompositeT>
	void add_composite(CompositeT& composite)
	{
		static_assert(is_composite<CompositeT>(),
				"Not a valid composite type");
		composite.manipulate(this);
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_connection(ArgT&& ... args)
	{
		this->add_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...));
	}

	/**
	 * If template unspecified, return EngineBase::Ptr
	 */
	template<typename EngineT = EngineBase>
	std::shared_ptr<EngineT> get_engine()
	{
		auto engine_ = std::dynamic_pointer_cast<EngineT>(this->engine);
		assert_throw_nullptr(engine_,
			NetworkException("get_engine()'s template type is incompatible"));
		return engine_;
	}

	/**
	 * If template unspecified, return DataManagerBase::Ptr
	 */
	template<typename DataManagerT = DataManagerBase>
	std::shared_ptr<DataManagerT> get_data_manager()
	{
		auto dataManager_ = std::dynamic_pointer_cast<DataManagerT>(this->dataManager);
		assert_throw_nullptr(dataManager_,
			NetworkException("get_data_manager()'s template type is incompatible"));
		return dataManager_;
	}

	/**************************************
	******* Training data management *********
	**************************************/
	virtual void load_input()
	{
		// FIXME for RNN this should be a sequence
		dataManager->upload_input(layers[0]->inValues[0]);
	}

	virtual void load_target()
	{
		dataManager->upload_target(lossLayer->targetValue[0]);
	}

	/**************************************
	******* Upload & exec instructions *********
	**************************************/
	virtual void upload(string methodName)
	{
		assert_throw(key_exists(networkMethodMap, methodName),
			NetworkException("no Network member method is associated with \"" + methodName + "\""));

		auto method = networkMethodMap[methodName];
		// call the member method
		// initialize, forward, backward, reset, etc.
		(this->*method)();
		this->routineMap[methodName] = engine->flush();
	}

	/**
	 * Compile all uploaded routines
	 */
	virtual void compile()
	{
		for (auto methodKey : this->routineMap)
			this->engine->compile(methodKey.second);
	}

	void execute(string methodName)
	{
		assert_throw(key_exists(routineMap, methodName),
			NetworkException("no Network Routine is associated with \"" + methodName + "\""));

		this->routineMap[methodName]->execute();
	}

	virtual void zero_clear() = 0;

	// TODO
	virtual void topological_sort()
	{
		throw UnimplementedException("topological sort");
	}


	// FIXME shouldn't be public
	vector<Layer::Ptr> layers;
	vector<Connection::Ptr> connections;
	vector<Component::Ptr> components;
	vector<ParamContainer::Ptr> paramContainers;

	LossLayerPtr lossLayer;

protected:
	/**
	 * Request DataManager to fill in input
	 */
	void forward()
	{
		check_initialized("forward");
		load_input();
		forward_impl();
	}

	virtual void forward_impl() = 0;

	/**
	 * Request DataManager to fill in target
	 */
	void backward()
	{
		check_initialized("backward");
		load_target();
		backward_impl();
	}

	virtual void backward_impl() = 0;

	void initialize()
	{
		assert_throw(!this->is_initialized,
			ComponentException("Network already initialized, can't init again unless reset()"));

		this->initialize_impl();

		this->is_initialized = true;
	}

	/**
	 * Subclasses should override this
	 */
	virtual void initialize_impl()
	{
		this->lossLayer = Layer::cast<LossLayer>(layers[layers.size() - 1]);

		assert_throw_nullptr(this->lossLayer,
			NetworkException("Last layer must be a LossLayer"));

		for (Component::Ptr c : this->components)
		{
			c->init_engine(this->engine);
			// call this at last after all other meta-params are set by init_XX()
			c->initialize();
		}
	}

protected:
	EngineBase::Ptr engine;

	DataManagerBase::Ptr dataManager;

	/**
	 * Initialization guard: should init only once except after reset()
	 */
	bool is_initialized = false;

	/**
	 * Contains all named routines.
	 * - "initialize_impl": initialization routine
	 * - "forward": forward propagation routine
	 * - "backward": backward propagation routine
	 */
	std::unordered_map<string, Routine::Ptr> routineMap;

	/**
	 * All named member methods
	 */
	std::unordered_map<string,
		decltype(&Network::forward)> networkMethodMap;

	/**
	 * Add to paramContainers only if 'component' is a subtype
	 */
	template<typename T>
	void check_add_param_container(T component)
	{
		ParamContainer::Ptr param = ParamContainer::upcast(component);
		if (param)
			paramContainers.push_back(param);
	}

	/**
	 * Exception helper
	 */
	void check_initialized(string msg)
	{
		assert_throw(this->is_initialized,
				NetworkException(msg + ": Network has not been initialized yet."));
	}
};

/**************************************
******* Feed-forward network *********
**************************************/
class ForwardNetwork : public Network
{
public:
	ForwardNetwork(EngineBase::Ptr engine_, DataManagerBase::Ptr dataManager_) :
		Network(engine_, dataManager_)
	{ }

	virtual ~ForwardNetwork() {};

// Unhide base class function with the same name but different signature
//	using Network::set_input;
//	using Network::set_target;

protected:
	virtual void initialize_impl()
	{
		Network::initialize_impl();
	}

	virtual void forward_impl()
	{
		for (Component::Ptr compon : this->components)
			compon->forward();
	}

	virtual void backward_impl()
	{
		for (int i = components.size() - 1; i >= 0; --i)
			components[i]->backward();
	}

	virtual void zero_clear()
	{
		for (Component::Ptr compon : this->components)
			compon->zero_clear();
	}
};

/**************************************
******* Recurrent Network *********
**************************************/
class RecurrentNetwork : public Network
{
public:
	RecurrentNetwork(EngineBase::Ptr engine_, DataManagerBase::Ptr dataManager_) :
		Network(engine_, dataManager_)
	{
		// defaults to 1, the most typical RNN
		set_max_temporal_skip(1);
	}

	virtual ~RecurrentNetwork() {};

	/*********** Network operations ***********/
	// FIXME
	virtual void initialize_impl()
	{
/*		assert_throw(input.size() == target.size(),
			NetworkException(
				"Input sequence must have the same length as the target sequence"));

		for (Layer::Ptr l : this->layers)
			l->init_history_length(this->input.size());*/

		for (Layer::Ptr layer : layers)
			layer->init_max_temporal_skip(this->maxTemporalSkip);

		Network::initialize_impl();
	}

	/**
	 * Call assemble() again to refresh maxTemporalSkip
	 * FIXME init_XX...
	 */
	virtual void set_max_temporal_skip(int maxTemporalSkip)
	{
		this->maxTemporalSkip = maxTemporalSkip;
	}

	/**
	 * First feeds forward in current time frame,
	 * then props to the next time frame
	 * FIXME need to for-loop over all frames in RNN forward
	 */
	virtual void forward_impl()
	{
		for (Component::Ptr compon : this->components)
		{
			Connection::Ptr conn = Component::cast<Connection>(compon);
			if (conn && key_exists(recurConnectionMap, conn))
			{
				int skip = recurConnectionMap[conn];
				if (frame >= skip)
					conn->forward(frame - skip, frame);
				else
					conn->prehistory_forward(
						// Ugly workaround for eclipse syntax highlighter
						std::static_pointer_cast<ParamContainer>(prehistoryLayerMap[conn->inLayer]),
						frame - skip, frame);
			}
			else
				compon->forward(frame, frame);
		}

		++ frame;
	}

	/**
	 * First back-props to the previous time point,
	 * then pass the gradient backward in current time.
	 */
	virtual void backward_impl()
	{
		-- frame;

		for (int i = components.size() - 1; i >= 0; --i)
		{
			Component::Ptr compon = components[i];
			Connection::Ptr conn = Component::cast<Connection>(compon);
			if (key_exists(recurConnectionMap, conn))
			{
				int skip = recurConnectionMap[conn];
				if (frame >= skip)
					conn->backward(frame, frame - skip);
				else
					conn->prehistory_backward(
						std::static_pointer_cast<ParamContainer>(prehistoryLayerMap[conn->inLayer]),
						frame, frame - skip);
			}
			else
				compon->backward(frame, frame);
		}

		for (Layer::Ptr layer : layers)
			layer->shift_back_gradient_window();
	}

	virtual void add_recurrent_connection(Connection::Ptr conn, int temporalSkip = 1)
	{
		assert_throw(
			maxTemporalSkip == Layer::UNLIMITED_TEMPORAL_SKIP
				|| temporalSkip <= maxTemporalSkip,
			NetworkException("temporalSkip should be <= maxTemporalSkip.\n"
					"Use set_max_temporal_skip() to change the upper limit.\n"
					"Then call assemble() again on the network"));

		components.push_back(Component::upcast(conn));
		connections.push_back(conn);
		this->check_add_param_container(conn);

		// Add the largest temporal skip for the layer
		auto prehistoryEntry = prehistoryLayerMap.find(conn->inLayer);
		auto& dummyRand = FakeRand::instance_prehistory();

		if (prehistoryEntry == prehistoryLayerMap.end())
		{
			auto h_0 = ParamContainer::make(temporalSkip);
			// FIXME
//			h_0->fill_rand(dummyRand);
			prehistoryLayerMap[conn->inLayer] = h_0;
			paramContainers.push_back(h_0);
		}
		else if (prehistoryEntry->second->size() < temporalSkip)
		{
			// FIXME prehistory must be able to resize
//			prehistoryEntry->second->resize(temporalSkip);
//			prehistoryEntry->second->fill_rand(dummyRand);
		}

		recurConnectionMap[conn] = temporalSkip;
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_recurrent_connection(ArgT&& ... args)
	{
		this->add_recurrent_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...));
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_recurrent_skip_connection(int temporalSkip, ArgT&& ... args)
	{
		this->add_recurrent_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...),
			temporalSkip);
	}

	virtual void zero_clear()
	{
		for (Component::Ptr compon : this->components)
			compon->zero_clear();

		// FIXME reset not written
//		for (auto& entry : prehistoryLayerMap)
//			entry.second->reset_gradients();

		frame = 0;
	}

	std::unordered_map<Layer::Ptr, ParamContainer::Ptr> prehistoryLayerMap;
	std::unordered_map<Connection::Ptr, int> recurConnectionMap;

protected:
	int frame = 0;
	int maxTemporalSkip = 1;
};

template<typename T>
typename enable_if<is_base_of<ForwardNetwork, T>::value, ostream>::type&
operator<<(ostream& os, T& net)
{
	os << "[ForwardNet\n";
	for (auto compon : net.components)
		os << "  " << string(*compon) << "\n";
	os << "]";
	return os;
}

template<typename T>
typename enable_if<is_base_of<RecurrentNetwork, T>::value, ostream>::type&
operator<<(ostream& os, T& net)
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

#endif /* NETWORK_H_ */
