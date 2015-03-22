/*
 * Eona Studio (c) 2015
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "component.h"
#include "layer.h"
#include "connection.h"

class Network
{
public:
	Network()
	{ }

	virtual ~Network() {}

	virtual void addLayer(LayerPtr) = 0;

	virtual void addConnection(ConnectionPtr) = 0;

	virtual void forward_prop() = 0;

	virtual void backward_prop() = 0;

	vector<LayerPtr> layers;
	vector<ConnectionPtr> connections;
	vector<ComponentPtr> components;
};

class ForwardNetwork : public Network
{
public:
	ForwardNetwork() { }

	~ForwardNetwork() {}

	virtual void addLayer(LayerPtr layer)
	{
		components.push_back(makeComponent(layer));
		layers.push_back(layer);
	}

	virtual void addConnection(ConnectionPtr conn)
	{
		components.push_back(makeComponent(conn));
		connections.push_back(conn);
	}

	virtual void forward_prop()
	{
		for (auto compon : this->components)
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
