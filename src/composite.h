/*
 * Eona Studio (c) 2015
 */

#ifndef COMPOSITE_H_
#define COMPOSITE_H_

#include "global_utils.h"
#include "layer.h"

class Network;

/**
 * Given an input layer, manipulate (add layers and connections)
 * to the network and return a new output layer.
 */
template<typename NetworkT>
class Composite
{
static_assert(std::is_base_of<Network, NetworkT>::value,
		"Composite<> template argument must be a subclass of Network type");

public:
	Composite(Layer::Ptr _inLayer) :
		inLayer(_inLayer)
	{
	}

	virtual ~Composite() =default;

	virtual void manipulate(Network *net)
	{
		NetworkT *netCast = dynamic_cast<NetworkT *>(net);

		if (netCast)
			this->_manipulate(netCast);
		else
			throw NetworkException("Composite is applied on a wrong Network type. ");
	}

	virtual Layer::Ptr& operator[](string name)
	{
		return this->_layerMap[name];
	}

	Layer::Ptr out_layer()
	{
		return this->outLayer;
	}

	/************************************/
	typedef shared_ptr<Composite<NetworkT> > Ptr;

	/**
	 * Initialize outLayer and other layers
	 * @return pointer
	 */
	template<typename CompositeT, typename ...ArgT>
	static Composite<NetworkT>::Ptr make(ArgT&& ... args)
	{
		auto compositePtr = static_cast<Composite<NetworkT>::Ptr>(
				std::make_shared<CompositeT>(
						std::forward<ArgT>(args) ...));

		compositePtr->outLayer = compositePtr->initialize_outlayer();
		compositePtr->initialize_layers(compositePtr->_layerMap);

		return compositePtr;
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
		comp.initialize_layers(comp._layerMap);

		return composite;
	}

	template<typename CompositeT>
	static shared_ptr<CompositeT> cast(Composite<NetworkT>::Ptr layer)
	{
		return std::dynamic_pointer_cast<CompositeT>(layer);
	}

protected:
	/**
	 * Composite logic goes here.
	 * Intended to work with network's "this" pointer
	 */
	virtual void _manipulate(NetworkT *net) = 0;

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
		return _layerMap[name];
	}

	LayerPtr inLayer, outLayer;

private:
	std::unordered_map<string, Layer::Ptr> _layerMap;
};

/**
 * Type trait
 */
template <class T>
std::true_type is_composite_impl(const Composite<T>* impl);

std::false_type is_composite_impl(...);

template <class Derived>
using is_composite =
		// simulate creation of a new Derived* pointer
    decltype(is_composite_impl(std::declval<Derived*>()));

#endif /* COMPOSITE_H_ */
