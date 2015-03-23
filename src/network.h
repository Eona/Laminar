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

	virtual void add_layer(LayerPtr) = 0;

	virtual void add_connection(ConnectionPtr) = 0;

	template<typename ConnectionT, typename ...ArgT>
	void add_new_connection(ArgT&& ... args)
	{
		this->add_connection(
			make_connection<ConnectionT>(std::forward<ArgT>(args)...));
	}

	virtual void forward_prop() = 0;

	virtual void backward_prop() = 0;

	virtual void reset() = 0;

	virtual void initialize() = 0;

	// TODO
	virtual void topological_sort()
	{
		throw UnimplementedException("topological sort");
	}

	vector<LayerPtr> layers;
	vector<ConnectionPtr> connections;
	vector<ComponentPtr> components;
};

class ForwardNetwork : public Network
{
public:
	ForwardNetwork() { }

	~ForwardNetwork() {}

	virtual void set_input(float input)
	{
		this->input = input;
	}

	virtual void set_target(float target)
	{
		this->target = target;
	}

	virtual void initialize()
	{
		layers[0]->inValue = this->input;
		this->lossLayer = cast_layer<LossLayer>(layers[layers.size() - 1]);
		if (lossLayer)
		{
			lossLayer->targetValue = this->target;
		}
		else
			throw NeuralException("Last layer must be a LossLayer");
	}

	virtual void add_layer(LayerPtr layer)
	{
		components.push_back(makeComponent(layer));
		layers.push_back(layer);
	}

	virtual void add_connection(ConnectionPtr conn)
	{
		components.push_back(makeComponent(conn));
		connections.push_back(conn);
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
		this->initialize();
	}

	// DUMMY
	float input = 0,
		target = 0;

	LossLayerPtr lossLayer;
};


/**
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
inline void gradient_check(ForwardNetwork& net,
		float perturb = 1e-3f, float percentTol = 0.5f)
{
	// for restoration
	float oldInput = net.input;

	for (ConnectionPtr conn : net.connections)
	{
		net.reset(); // refresh network

		auto linearConn = cast_connection<LinearConnection>(conn);
		auto constConn = cast_connection<ConstantConnection>(conn);
		if (linearConn)
		{
			net.forward_prop();
			net.backward_prop();
			float analyticGrad = linearConn->gradient;

			// perturb the parameter and run again
			net.reset();
			linearConn->param -= perturb;
			net.forward_prop();
			float outValMinus = net.lossLayer->outValue;
			net.reset();
			linearConn->param += perturb;
			net.forward_prop();
			float outValPlus = net.lossLayer->outValue;
			float numericGrad = (outValPlus - outValMinus) / (2.0 * perturb);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"analytic != numeric", "gradient check success");
		}
		else if (constConn) { }
	}
}

ostream& operator<<(ostream& os, ForwardNetwork& layer)
{
	os << "[ForwardNet\n";
	for (auto compon : layer.components)
		os << "  " << compon->str() << "\n";
	os << "]";
	return os;
}

#endif /* NETWORK_H_ */
