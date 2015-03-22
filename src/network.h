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

	virtual void forward_prop() = 0;

	virtual void backward_prop() = 0;

	vector<Component*> components;
};

class ForwardNetwork : public Network
{
public:
	ForwardNetwork(std::initializer_list<Component*> _components) :
		Network(_components)
	{}

	~ForwardNetwork() {}

	virtual void forward_prop()
	{
		for (Component *compon : this->components)
			compon->forward();
	}

	virtual void backward_prop()
	{
		for (int i = components.size() - 1; i >= 0; --i)
			components[i]->backward();
	}
};

ostream& operator<<(ostream& os, ForwardNetwork& layer)
{
	os << "[ForwardNet\n";
	for (auto compon : layer.components)
		os << "  " << compon->str() << "\n";
	os << "]";
	return os;
}

#endif /* NETWORK_H_ */
