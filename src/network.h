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

	virtual void set_input(vector<float>& input)
	{
		this->input = input;
	}
	virtual void set_input(vector<float>&& input)
	{
		this->input = input;
	}

	virtual void set_target(vector<float>& target)
	{
		this->target = target;
	}
	virtual void set_target(vector<float>&& target)
	{
		this->target = target;
	}

	virtual void add_layer(LayerPtr layer)
	{
		components.push_back(Component::upcast(layer));
		layers.push_back(layer);

		this->check_add_param_container(layer);
	}

	virtual void add_connection(ConnectionPtr conn)
	{
		components.push_back(Component::upcast(conn));
		connections.push_back(conn);

		this->check_add_param_container(conn);
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_connection(ArgT&& ... args)
	{
		this->add_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...));
	}

	virtual void forward_prop() = 0;

	virtual void backward_prop() = 0;

	virtual void reset() = 0;

	virtual void assemble() = 0;

	// TODO
	virtual void topological_sort()
	{
		throw UnimplementedException("topological sort");
	}

	vector<Layer::Ptr> layers;
	vector<Connection::Ptr> connections;
	vector<Component::Ptr> components;
	vector<ParamContainer::Ptr> paramContainers;

	LossLayerPtr lossLayer;

	vector<float> input, target;

protected:
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

class ForwardNetwork : public Network
{
public:
	ForwardNetwork() :
		Network()
	{ }

	~ForwardNetwork() {}

	// Unhide base class function with the same name but different signature
	using Network::set_input;
	using Network::set_target;

	virtual void set_input(float input)
	{
		Network::set_input(vector<float> {input});
	}

	virtual void set_target(float target)
	{
		Network::set_target(vector<float> {target});
	}

	virtual void assemble()
	{
		layers[0]->inValues = this->input;
		this->lossLayer = Layer::cast<LossLayer>(layers[layers.size() - 1]);
		if (lossLayer)
		{
			lossLayer->targetValue = this->target;
		}
		else
			throw NetworkException("Last layer must be a LossLayer");
	}

	virtual void forward_prop()
	{
		for (ComponentPtr compon : this->components)
			compon->forward();
	}

	virtual void backward_prop()
	{
		for (int i = components.size() - 1; i >= 0; --i)
			components[i]->backward();
	}

	virtual void reset()
	{
		for (ComponentPtr compon : this->components)
			compon->reset();
		this->assemble();
	}
};


class RecurrentNetwork : public Network
{
public:
	RecurrentNetwork() :
		Network()
	{ }

	~RecurrentNetwork() {}

	virtual void assemble()
	{
		if (input.size() != target.size())
			throw NetworkException(
					"Input sequence must have the same length as the target sequence");

		layers[0]->inValues = this->input;
		this->lossLayer = Layer::cast<LossLayer>(layers[layers.size() - 1]);
		if (lossLayer)
		{
			lossLayer->targetValue = this->target;
		}
		else
			throw NetworkException("Last layer must be a LossLayer");
	}

	// TODO check last timestamp (cannot forward beyond)
	// TODO add h0 as another parameter (first layer has no previous time)
	/**
	 * First feeds forward in current time frame,
	 * then props to the next time frame
	 */
	virtual void forward_prop()
	{
//		DEBUG_MSG("Forward frame", frame);

		for (ComponentPtr compon : this->components)
		{
			compon->forward(frame, frame);
		}

		// Recurrent forward prop
		for (ConnectionPtr conn : this->recurConnections)
			conn->forward(frame, frame + 1);

		++ frame;
	}

	// TODO check last timestamp (cannot forward beyond)
	// TODO compute gradient for h0
	/**
	 * First back-props to the previous time point,
	 * then pass the gradient backward in current time.
	 */
	virtual void backward_prop()
	{
		-- frame;
//		DEBUG_MSG("Backward frame", frame);

		for (int i = recurConnections.size() - 1; i >= 0; --i)
			recurConnections[i]->backward(frame + 1, frame);

		for (LayerPtr layer : layers)
			layer->shiftBackGradientWindow();

		for (int i = components.size() - 1; i >= 0; --i)
			components[i]->backward(frame, frame);
	}

	virtual void add_recurrent_connection(ConnectionPtr conn)
	{
		recurConnections.push_back(conn);
		connections.push_back(conn);
		this->check_add_param_container(conn);
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_recurrent_connection(ArgT&& ... args)
	{
		this->add_recurrent_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...));
	}

	virtual void reset()
	{
		for (ComponentPtr compon : this->components)
			compon->reset();
		for (ConnectionPtr conn : this->recurConnections)
			conn->reset();

		frame = 0;

		this->assemble();
	}

	vector<ConnectionPtr> recurConnections;
	int frame = 0;
};

template<typename T>
typename enable_if<is_base_of<ForwardNetwork, T>::value, ostream>::type&
operator<<(ostream& os, T& net)
{
	os << "[ForwardNet\n";
	for (auto compon : net.components)
		os << "  " << compon->str() << "\n";
	os << "]";
	return os;
}

template<typename T>
typename enable_if<is_base_of<RecurrentNetwork, T>::value, ostream>::type&
operator<<(ostream& os, T& net)
{
	os << "[RecurrentNet\n";
	for (auto compon : net.components)
		os << "  " << compon->str() << "\n";
	os << " " << "recurrent connections:\n";
	for (auto recConn : net.recurConnections)
		os << "  " << recConn->str() << "\n";
	os << "]";
	return os;
}

#endif /* NETWORK_H_ */
