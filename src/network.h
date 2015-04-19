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
		networkMethodMap["zero_clear"] = &Network::zero_clear;
		networkMethodMap["load_input"] = &Network::load_input;
		networkMethodMap["load_target"] = &Network::load_target;
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
	/**
	 * forward() loads input every time
	 */
	virtual void load_input() = 0;

	/**
	 * backward() loads target every time
	 */
	virtual void load_target() = 0;

	/**************************************
	******* Upload & exec instructions *********
	**************************************/
	/**
	 * The method must have been registered to networkMethodMap in ctor
	 * For the first time, the method will be run (instructions uploaded),
	 * and the generated Routine will be compiled and executed
	 * The next time it's called, the compiled routine will be executed directly.
	 * @param methodName
	 */
	void execute(string methodName)
	{
		assert_throw(key_exists(networkMethodMap, methodName),
			NetworkException("no Network member method is associated with \"" + methodName + "\""));

		// if the instructions haven't been generated
		if (!key_exists(routineMap, methodName))
		{
			this->compile_helper(methodName);
		}

		this->routineMap[methodName]->execute();
	}

	/**
	 * Recompile an executed method
	 */
	void recompile(string methodName)
	{
		assert_throw(key_exists(networkMethodMap, methodName),
			NetworkException("no Network member method is associated with \"" + methodName + "\""));

		assert_throw(key_exists(routineMap, methodName),
			NetworkException(methodName + " has never been compiled."));

		this->compile_helper(methodName);
	}

	virtual void zero_clear() = 0;

	// WISHLIST
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
	 * Compile the method and store it to routineMap
	 * Doesn't do any status check
	 * @param methodName
	 */
	void compile_helper(string methodName)
	{
		auto method = networkMethodMap[methodName];
		// call the member method
		// initialize, forward, backward, reset, etc.
		(this->*method)();
		auto routine = engine->flush();
		engine->compile(routine);
		this->routineMap[methodName] = routine;
	}

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
	 * The function should be called *after* initialization
	 */
	void check_initialized(string msg)
	{
		assert_throw(this->is_initialized,
			NetworkException(msg + ": Network has not been initialized yet."));
	}

	/**
	 * Exception helper
	 * The function should be called *before* initialization
	 */
	void check_uninitialized(string msg)
	{
		assert_throw(!this->is_initialized,
			NetworkException(msg + " should be called before Network initialization."));
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

	virtual void load_input()
	{
		dataManager->upload_input(layers[0]->in_value(0));
	}

	virtual void load_target()
	{
		dataManager->upload_target(lossLayer->target_value(0));
	}

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
	RecurrentNetwork(EngineBase::Ptr engine_, DataManagerBase::Ptr dataManager_,
			int historyLength_, int maxTemporalSkip_ = 1) :
		Network(engine_, dataManager_),
		historyLength(historyLength_),
		maxTemporalSkip(maxTemporalSkip_)
	{ }

	virtual ~RecurrentNetwork() {};

	virtual void init_max_temporal_skip(int maxTemporalSkip)
	{
		Network::check_uninitialized("init_max_temporal_skip");
		this->maxTemporalSkip = maxTemporalSkip;
	}

	int max_temporal_skip()
	{
		return this->maxTemporalSkip;
	}

	virtual void init_history_length(int historyLength)
	{
		Network::check_uninitialized("init_history_length");
		this->historyLength = historyLength;
	}

	int history_length()
	{
		return this->historyLength;
	}

	virtual void add_recurrent_connection(Connection::Ptr conn, int temporalSkip = 1)
	{
		assert_throw(
			maxTemporalSkip == Layer::UNLIMITED_TEMPORAL_SKIP
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

	/**************************************
	******* Training data management *********
	**************************************/
	/**
	 * Loads a sequence of input
	 */
	virtual void load_input()
	{
		dataManager->start_new_sequence();
		for (int frame = 0; frame < this->historyLength; ++frame)
			dataManager->upload_input(layers[0]->in_value(frame));
	}

	/**
	 * Loads a sequence of target
	 */
	virtual void load_target()
	{
		dataManager->start_new_sequence();
		for (int frame = 0; frame < this->historyLength; ++frame)
			dataManager->upload_target(lossLayer->target_value(frame));
	}

	virtual void zero_clear()
	{
		for (Component::Ptr compon : this->components)
			compon->zero_clear();

		for (auto& entry : prehistoryMap)
			entry.second->clear_gradients();

//		frame = 0;
	}

protected:
	/*********** Network operations ***********/
	virtual void initialize_impl()
	{
		// initialize prehistory ParamContainers
		this->init_prehistory_params();

		for (Layer::Ptr l : this->layers)
			l->init_history_length(this->historyLength);

		for (Layer::Ptr layer : layers)
			layer->init_max_temporal_skip(this->maxTemporalSkip);

		Network::initialize_impl();
	}

	/**
	 * First feeds forward in current time frame,
	 * then props to the next time frame
	 */
	virtual void forward_impl()
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
	virtual void backward_impl()
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

			// FIXME FIXME: dim of prehistory should be { layerDim * batchSize }
				valuePtr = Tensor::make(engine, vec_augment(layer->dim(), 1));
				gradPtr = Tensor::make(engine, vec_augment(layer->dim(), 1));

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
