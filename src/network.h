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
		assert_throw<NetworkException>(engine == dataManager->get_engine(),
				"DataManager has a different engine");

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
		LMN_STATIC_ASSERT(is_composite<CompositeT>(),
				"Not a valid composite type");

		composite->manipulate(this);
	}

	template<typename CompositeT>
	void add_composite(CompositeT& composite)
	{
		LMN_STATIC_ASSERT(is_composite<CompositeT>(),
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
		assert_throw_nullptr<NetworkException>(engine_,
			"get_engine()'s template type is incompatible");
		return engine_;
	}

	/**
	 * If template unspecified, return DataManagerBase::Ptr
	 */
	template<typename DataManagerT = DataManagerBase>
	std::shared_ptr<DataManagerT> get_data_manager()
	{
		auto dataManager_ = std::dynamic_pointer_cast<DataManagerT>(this->dataManager);
		assert_throw_nullptr<NetworkException>(dataManager_,
			"get_data_manager()'s template type is incompatible");
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
		assert_throw<NetworkException>(key_exists(networkMethodMap, methodName),
			"no Network member method is associated with \"" + methodName + "\"");

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
		assert_throw<NetworkException>(key_exists(networkMethodMap, methodName),
			"no Network member method is associated with \"" + methodName + "\"");

		assert_throw<NetworkException>(key_exists(routineMap, methodName),
				methodName + " has never been compiled.");

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
	 * Request DataManager to fill in input and target
	 */
	void forward()
	{
		check_initialized("forward");
		load_input();
		load_target();
		forward_impl();
	}

	virtual void forward_impl() = 0;

	void backward()
	{
		check_initialized("backward");
		backward_impl();
	}

	virtual void backward_impl() = 0;

	void initialize()
	{
		assert_throw<NetworkException>(!this->is_initialized,
			"Network already initialized, can't init again unless reset()");

		this->initialize_impl();

		this->is_initialized = true;
	}

	/**
	 * Subclasses should override this
	 */
	virtual void initialize_impl()
	{
		this->lossLayer = Layer::cast<LossLayer>(layers[layers.size() - 1]);

		assert_throw_nullptr<NetworkException>(this->lossLayer,
				"Last layer must be a LossLayer");

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
		assert_throw<NetworkException>(this->is_initialized,
			msg + ": Network has not been initialized yet.");
	}

	/**
	 * Exception helper
	 * The function should be called *before* initialization
	 */
	void check_uninitialized(string msg)
	{
		assert_throw<NetworkException>(!this->is_initialized,
			msg + " should be called before Network initialization.");
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

template<typename T>
typename std::enable_if<std::is_base_of<ForwardNetwork, T>::value, std::ostream>::type&
operator<<(std::ostream& os, T& net)
{
	os << "[ForwardNet\n";
	for (auto compon : net.components)
		os << "  " << string(*compon) << "\n";
	os << "]";
	return os;
}

#endif /* NETWORK_H_ */
