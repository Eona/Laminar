/*
 * Eona Studio (c) 2015
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "component.h"
#include "composite.h"
#include "layer.h"
#include "bias_layer.h"
#include "connection.h"
#include "engine/engine.h"
#include "engine/tensor.h"
#include "engine/data_manager.h"

class Network
{
public:
	Network(EngineBase::Ptr engine_, DataManagerBase::Ptr dataManager_) :
		engine(engine_),
		dataManager(dataManager_),
		initGuard("Network")
	{
		LMN_ASSERT_THROW(engine == dataManager->get_engine(),
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
		initGuard.assert_before_initialize("add_layer");

		this->components.push_back(Component::upcast(layer));
		this->layers.push_back(layer);

		// WISHLIST any better idea to treat biases?
		auto biasLayer = Layer::cast<BiasLayer>(layer);
		if (biasLayer)
			this->biases.push_back(biasLayer);

		this->check_add_param_container(layer);
	}

	virtual void add_connection(Connection::Ptr conn)
	{
		initGuard.assert_before_initialize("add_connection");

		this->components.push_back(Component::upcast(conn));
		this->connections.push_back(conn);

		this->check_add_param_container(conn);
	}

	/**
	 * Convenience method, same as:
	 * 1) Construct a new BiasLayer
	 * 2) Add it to this network
	 * 3) Add a FullConnection from the bias to its output layer
	 * WISHLIST make the interface work with ConvNet bias
	 */
	virtual void new_bias_layer(Layer::Ptr layer)
	{
		initGuard.assert_before_initialize("new_bias_layer");

		// WISHLIST topological sort can figure this out, no need to check
		LMN_ASSERT_THROW(!vec_contains(this->layers, layer),
			NetworkException("BiasLayer must be added before its output layer"));

		auto biasLayer = Layer::make<BiasLayer>();
		this->add_layer(biasLayer);
		this->new_connection<FullConnection>(biasLayer, layer);
	}

	/**
	 * Cannot add a composite that requires a more specialized network.
	 */
	template<typename CompositeT>
	void add_composite(std::shared_ptr<CompositeT> composite)
	{
		LMN_STATIC_ASSERT(is_composite<CompositeT>(),
				"Not a valid composite type");

		initGuard.assert_before_initialize("add_composite");

		composite->manipulate(this);
	}

	template<typename CompositeT>
	void add_composite(CompositeT& composite)
	{
		LMN_STATIC_ASSERT(is_composite<CompositeT>(),
				"Not a valid composite type");

		initGuard.assert_before_initialize("add_composite");
		composite.manipulate(this);
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_connection(ArgT&& ... args)
	{
		initGuard.assert_before_initialize("new_connection");

		this->add_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...));
	}

	/**************************************
	******* Getter methods *********
	**************************************/
	/**
	 * If template unspecified, return EngineBase::Ptr
	 */
	template<typename EngineT = EngineBase>
	std::shared_ptr<EngineT> get_engine()
	{
		auto engine_ =
				std::dynamic_pointer_cast<EngineT>(this->engine);
		LMN_ASSERT_NULLPTR(engine_,
			NetworkException("get_engine()'s template type is incompatible"));
		return engine_;
	}

	/**
	 * If template unspecified, return DataManagerBase::Ptr
	 */
	template<typename DataManagerT = DataManagerBase>
	std::shared_ptr<DataManagerT> get_data_manager()
	{
		auto dataManager_ =
				std::dynamic_pointer_cast<DataManagerT>(this->dataManager);
		LMN_ASSERT_NULLPTR(dataManager_,
			NetworkException("get_data_manager()'s template type is incompatible"));
		return dataManager_;
	}

	vector<ParamContainer::Ptr> get_param_containers()
	{
		initGuard.assert_after_initialize("get_param_containers");
		return this->paramContainers;
	}

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
		if (methodName == "initialize")
			initGuard.initialize();
		else
			initGuard.assert_after_initialize("execute " + methodName);

		LMN_ASSERT_THROW(key_exists(networkMethodMap, methodName),
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
		LMN_ASSERT_THROW(key_exists(networkMethodMap, methodName),
			NetworkException("no Network member method is associated with \"" + methodName + "\""));

		LMN_ASSERT_THROW(key_exists(routineMap, methodName),
			NetworkException(methodName + " has never been compiled."));

		initGuard.assert_after_initialize("recompile " + methodName);

		this->compile_helper(methodName);
	}

	// WISHLIST
	virtual void topological_sort()
	{
		initGuard.assert_before_initialize("topological_sort");
		throw UnimplementedException("topological sort");
	}

	/************************************/
	TYPEDEF_PTR(Network);

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(Network)

	/**
	 * Down cast NetworkPtr to a specific network type
	 */
	GEN_DOWN_CAST_STATIC_MEMBER(Network)


	// FIXME shouldn't be public
	vector<Layer::Ptr> layers;
	vector<BiasLayer::Ptr> biases;
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

	/**************************************
	******* Training logic *********
	**************************************/
	/**
	 * Asks dataManager to fill in input
	 */
	virtual void load_input() = 0;

	/**
	 * Asks dataManager to fill in target
	 */
	virtual void load_target() = 0;

	/**
	 * Main forward propagation logic
	 */
	virtual void forward() = 0;

	/**
	 * Main backward propagation logic
	 */
	virtual void backward() = 0;

	/**
	 * Clears all layer in/out values/gradients and parameter gradients
	 */
	virtual void zero_clear() = 0;

	/**
	 * Subclasses should override this
	 */
	virtual void initialize()
	{
		this->lossLayer = Layer::cast<LossLayer>(layers[layers.size() - 1]);
		LMN_ASSERT_NULLPTR(this->lossLayer,
				NetworkException("Last layer must be a LossLayer"));

		// WISHLIST any better way to init bias?
		for (BiasLayer::Ptr b : this->biases)
			b->init_batch_size(this->dataManager->batch_size());

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

	InitializeGuard<NetworkException> initGuard;

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

	/************************************/
	TYPEDEF_PTR(ForwardNetwork);

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(ForwardNetwork)

protected:
	virtual void initialize()
	{
		Network::initialize();
	}

	virtual void load_input()
	{
		dataManager->upload_input(layers[0]->in_value(0));
	}

	virtual void load_target()
	{
		dataManager->upload_target(lossLayer->target_value(0));
	}

	virtual void forward()
	{
		for (Component::Ptr compon : this->components)
			compon->forward();
	}

	virtual void backward()
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
