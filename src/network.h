/*
 * Eona Studio (c) 2015
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.h"
#include "connection.h"

class Network
{
public:
	Network(std::initializer_list<Component*> _components) :
		components(_components)
	{
	}

	virtual ~Network() {}

	vector<Component*> components;
};

class ForwardNetwork : public Network
{
public:
	ForwardNetwork(std::initializer_list<Component*> _components) :
		Network(_components)
	{}

	~ForwardNetwork() {}
};

ostream& operator<<(ostream& os, ForwardNetwork& layer)
{
	os << "[ForwardNet\n";
	for (auto compon : layer.components)
		os << compon->str() << "\n";
	os << "]";
	return os;
}

#endif /* NETWORK_H_ */
