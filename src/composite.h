/*
 * Eona Studio (c) 2015
 */

#ifndef COMPOSITE_H_
#define COMPOSITE_H_

#include "global_utils.h"
#include "layer.h"

class Network;

template<typename NetworkT>
class Composite
{
static_assert(std::is_base_of<Network, NetworkT>::value,
		"Composite<> template argument must be a subclass of Network type");

public:
	Composite(Layer::Ptr _inLayer, Layer::Ptr _outLayer) :
		inLayer(bool(_inLayer) ?
				_inLayer : initialize_inlayer_if_null()),
		outLayer(bool(_outLayer) ?
				_outLayer : initialize_outlayer_if_null())
	{
		this->initialize_layers(this->_layerMap);
	}

	/**
	 * Pass null pointers for in/outLayer
	 */
	Composite() :
		Composite(Layer::Ptr(), Layer::Ptr())
	{
	}

	virtual ~Composite() =default;

	/**
	 * Will be called in constructor
	 */
	virtual void initialize_layers(
			std::unordered_map<string, Layer::Ptr>& layerMap) = 0;

	/**
	 * Will be called if inLayer is not specified
	 */
	virtual Layer::Ptr initialize_inlayer_if_null() = 0;

	/**
	 * Will be called if outLayer is not specified
	 */
	virtual Layer::Ptr initialize_outlayer_if_null() = 0;

	/**
	 * Composite logic goes here.
	 * Intended to work with network's "this" pointer
	 */
	virtual void manipulate(NetworkT *net) = 0;

	virtual Layer::Ptr& operator[](string name)
	{
		return this->_layerMap[name];
	}

	/************************************/
	typedef shared_ptr<Composite<NetworkT> > Ptr;

	static shared_ptr<Composite<NetworkT> > cast(Ptr composite)
	{
		return std::dynamic_pointer_cast<Composite<NetworkT> >(composite);
	}

protected:
	Layer::Ptr get_layer(string name)
	{
		return _layerMap[name];
	}

	LayerPtr inLayer, outLayer;

private:
	std::unordered_map<string, Layer::Ptr> _layerMap;
};

#endif /* COMPOSITE_H_ */
