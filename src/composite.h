/*
 * Eona Studio (c) 2015
 */

#ifndef COMPOSITE_H_
#define COMPOSITE_H_

#include "layer.h"
#include "utils/global_utils.h"

class Network;

/**
 * Given an input layer, manipulate (add layers and connections)
 * to the network and return a new output layer.
 */
template<typename NetworkT>
class Composite
{
LMN_STATIC_ASSERT((std::is_base_of<Network, NetworkT>::value),
		"Composite<> template argument must be a subclass of Network type");

public:
	Composite(Layer::Ptr inLayer_) :
		inLayer(inLayer_)
	{
	}

	virtual ~Composite() {};

	virtual void manipulate(Network *net)
	{
		NetworkT *netCast = dynamic_cast<NetworkT *>(net);

		if (netCast)
			this->manipulate_impl(netCast);
		else
			throw NetworkException("Composite is applied on a wrong Network type.");
	}

	virtual Layer::Ptr& operator[](string name)
	{
		return this->layerMap[name];
	}

	Layer::Ptr out_layer()
	{
		return this->outLayer;
	}

	/************************************/
	TYPEDEF_PTR(Composite<NetworkT>);

	/**
	 * Initialize outLayer and other layers
	 * @return pointer
	 */
	template<typename CompositeT, typename ...ArgT>
	static Composite<NetworkT>::Ptr make(ArgT&& ... args)
	{
		auto compPtr = std::static_pointer_cast<Composite<NetworkT>>(
			std::make_shared<CompositeT>( std::forward<ArgT>(args) ...));

		compPtr->outLayer = compPtr->initialize_outlayer();
		compPtr->initialize_layers(compPtr->layerMap);

		return compPtr;
	}

	/**
	 * @return object
	 */
	template<typename CompositeT, typename ...ArgT>
	static CompositeT create(ArgT&& ... args)
	{
		CompositeT composite(std::forward<ArgT>(args) ...);

		// WARNING workaround: if operate on object directly, all methods of the 
		// subclass must be public, otherwise compiler throws inaccessible error. 
		// Only reference and pointer types are polymorphic.
		Composite<NetworkT>& comp = composite;

		comp.outLayer = comp.initialize_outlayer();
		comp.initialize_layers(comp.layerMap);

		return composite;
	}

	template<typename CompositeT>
	static std::shared_ptr<CompositeT> cast(Composite<NetworkT>::Ptr layer)
	{
		return std::dynamic_pointer_cast<CompositeT>(layer);
	}

protected:
	/**
	 * Composite logic goes here.
	 * Intended to work with network's "this" pointer
	 * @param inLayer dimension
	 */
	virtual void manipulate_impl(NetworkT *net) = 0;

	/**
	 * Will be called in static ::make
	 */
	virtual void initialize_layers(
		std::unordered_map<string, Layer::Ptr>& layerMap) = 0;

	/**
	 * Will be called in static ::make
	 */
	virtual Layer::Ptr initialize_outlayer() = 0;

	Layer::Ptr get_layer(string name)
	{
		return layerMap[name];
	}

	LayerPtr inLayer, outLayer;

private:
	std::unordered_map<string, Layer::Ptr> layerMap;
};

/**
 * Type trait
 */
GEN_IS_DERIVED_TEMPLATE_TRAIT(is_composite, Composite);

#endif /* COMPOSITE_H_ */
