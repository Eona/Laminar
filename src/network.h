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

	virtual void add_layer(LayerPtr layer)
	{
		components.push_back(make_component(layer));
		layers.push_back(layer);
	}

	virtual void add_connection(ConnectionPtr conn)
	{
		components.push_back(make_component(conn));
		connections.push_back(conn);
	}

	template<typename ConnectionT, typename ...ArgT>
	void add_new_connection(ArgT&& ... args)
	{
		this->add_connection(
			make_connection<ConnectionT>(
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

	vector<LayerPtr> layers;
	vector<ConnectionPtr> connections;
	vector<ComponentPtr> components;

	LossLayerPtr lossLayer;
};

class ForwardNetwork : public Network
{
public:
	ForwardNetwork() :
		Network()
	{ }

	~ForwardNetwork() {}

	virtual void set_input(float input)
	{
		this->input = input;
	}

	virtual void set_target(float target)
	{
		this->target = target;
	}

	virtual void assemble()
	{
		layers[0]->inValue[0] = this->input;
		this->lossLayer = cast_layer<LossLayer>(layers[layers.size() - 1]);
		if (lossLayer)
		{
			lossLayer->targetValue[0] = this->target;
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

	// DUMMY
	float input = 0,
		target = 0;
};


class RecurrentNetwork : public Network
{
public:
	RecurrentNetwork() :
		Network()
	{ }

	~RecurrentNetwork() {}

	virtual void set_input(vector<float> input)
	{
		this->input = input;
	}

	virtual void set_target(vector<float> target)
	{
		this->target = target;
	}

	virtual void assemble()
	{
		if (input.size() != target.size())
			throw NetworkException("");

		layers[0]->inValue[0] = this->input[0];
		this->lossLayer = cast_layer<LossLayer>(layers[layers.size() - 1]);
		if (lossLayer)
		{
			lossLayer->targetValue[0] = this->target[0];
		}
		else
			throw NetworkException("Last layer must be a LossLayer");
	}

	virtual void forward_prop()
	{
		// Recurrent forward prop
		for (ConnectionPtr conn : this->recurConnections)
			conn->forward(time, time + 1);

		for (ComponentPtr compon : this->components)
		{
			auto conn = cast_component<Connection>(compon);
			if (conn)
				conn->forward(time, time);
			else
				compon->forward();
		}

		++ time;
	}

	virtual void backward_prop()
	{
		for (int i = components.size() - 1; i >= 0; --i)
			components[i]->backward();
	}

	virtual void add_recurrent_connection(ConnectionPtr conn)
	{
		recurConnections.push_back(conn);
	}

	virtual void reset()
	{
		for (ComponentPtr compon : this->components)
			compon->reset();
		for (ConnectionPtr conn : this->recurConnections)
			conn->reset();

		time = 0;

		this->assemble();
	}

	vector<ConnectionPtr> recurConnections;
	int time = 0;
	vector<float> input, target;
};

/**
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
inline void gradient_check(ForwardNetwork& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	/****** perturb parameters matrices stored in connections ******/
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
			float oldParam = linearConn->param; // for restoration

			// perturb the parameter and run again
			net.reset();
			linearConn->param = oldParam - perturb;
			net.forward_prop();
			float outValMinus = net.lossLayer->outValue[0];

			net.reset();
			linearConn->param = oldParam + perturb;
			net.forward_prop();
			float outValPlus = net.lossLayer->outValue[0];

			float numericGrad = (outValPlus - outValMinus) / (2.0 * perturb);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"param analytic != numeric", "param gradcheck pass");

			linearConn->param = oldParam;
		}
		else if (constConn) { }
	}

	/****** perturb the input ******/
	float oldInput = net.input; // for restoration

	net.reset();
	net.forward_prop();
	net.backward_prop();
	float analyticGrad = net.layers[0]->inGradient[0];

	net.set_input(oldInput - perturb);
	net.reset();
	net.forward_prop();
	float outValMinus = net.lossLayer->outValue[0];

	net.set_input(oldInput + perturb);
	net.reset();
	net.forward_prop();
	float outValPlus = net.lossLayer->outValue[0];

	float numericGrad = (outValPlus - outValMinus) / (2.0 * perturb);

	assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
			"input analytic != numeric", "input gradcheck pass");
}


/**
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
inline void gradient_check(RecurrentNetwork& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	/****** perturb parameters matrices stored in connections ******/
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
			float oldParam = linearConn->param; // for restoration

			// perturb the parameter and run again
			net.reset();
			linearConn->param = oldParam - perturb;
			net.forward_prop();
			float outValMinus = net.lossLayer->outValue[0];

			net.reset();
			linearConn->param = oldParam + perturb;
			net.forward_prop();
			float outValPlus = net.lossLayer->outValue[0];

			float numericGrad = (outValPlus - outValMinus) / (2.0 * perturb);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"param analytic != numeric", "param gradcheck pass");

			linearConn->param = oldParam;
		}
		else if (constConn) { }
	}

	/****** perturb the input ******/
	float oldInput = net.input[0]; // for restoration

	net.reset();
	net.forward_prop();
	net.backward_prop();
	float analyticGrad = net.layers[0]->inGradient[0];

	// TODO
//	net.set_input(oldInput - perturb);
	net.reset();
	net.forward_prop();
	float outValMinus = net.lossLayer->outValue[0];

	// TODO
//	net.set_input(oldInput + perturb);
	net.reset();
	net.forward_prop();
	float outValPlus = net.lossLayer->outValue[0];

	float numericGrad = (outValPlus - outValMinus) / (2.0 * perturb);

	assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
			"input analytic != numeric", "input gradcheck pass");
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
