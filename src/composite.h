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

	/**
	 * Composite logic goes here.
	 * Intended to work with network's "this" pointer
	 */
	virtual void manipulate(NetworkT *net) = 0;

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
	 */
	template<typename CompositeT, typename ...ArgT>
	static Composite<NetworkT>::Ptr make(ArgT&& ... args)
	{
		auto compos = static_cast<Composite<NetworkT>::Ptr>(
				std::make_shared<CompositeT>(
						std::forward<ArgT>(args) ...));

		compos->outLayer = compos->initialize_outlayer();
		compos->initialize_layers(compos->_layerMap);

		return compos;
	}

	template<typename CompositeT>
	static shared_ptr<CompositeT> cast(Composite<NetworkT>::Ptr layer)
	{
		return std::dynamic_pointer_cast<CompositeT>(layer);
	}

protected:
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

#endif /* COMPOSITE_H_ */
