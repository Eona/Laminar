/*
 * Eona Studio (c) 2015
 */

#ifndef COMPOSITE_H_
#define COMPOSITE_H_

#include "global_utils.h"

class Network;

template<typename NetworkT>
class Composite
{
	static_assert(std::is_base_of<Network, NetworkT>::value,
			"Composite<> template argument must be a subclass of Network type");

	virtual ~Composite() =default;

	/**
	 * Intended to work with network's "this" pointer
	 */
	virtual void manipulate(NetworkT *net) = 0;

	/************************************/
	typedef shared_ptr<Composite<NetworkT> > Ptr;

	static shared_ptr<Composite<NetworkT> > cast(Ptr composite)
	{
		return std::dynamic_pointer_cast<Composite<NetworkT> >(composite);
	}
};

#endif /* COMPOSITE_H_ */
