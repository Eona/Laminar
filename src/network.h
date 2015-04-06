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
	{
		// defaults to 1, the most typical RNN
		set_max_temporal_skip(1);
	}

	~RecurrentNetwork() {}

	/*********** Network operations ***********/
	virtual void assemble()
	{
		assert_throw(input.size() == target.size(),
			NetworkException(
				"Input sequence must have the same length as the target sequence"));

		for (LayerPtr layer : layers)
			layer->set_max_temporal_skip(this->maxTemporalSkip);

		layers[0]->inValues = this->input;
		this->lossLayer = Layer::cast<LossLayer>(layers[layers.size() - 1]);
		if (lossLayer)
		{
			lossLayer->targetValue = this->target;
		}
		else
			throw NetworkException("Last layer must be a LossLayer");
	}

	/**
	 * Call assemble() again to refresh maxTemporalSkip
	 */
	virtual void set_max_temporal_skip(int maxTemporalSkip)
	{
		this->maxTemporalSkip = maxTemporalSkip;
	}

	/**
	 * First feeds forward in current time frame,
	 * then props to the next time frame
	 */
	virtual void forward_prop()
	{
		// Recurrent forward prop
		for (auto& connInfo : this->recurConnectionInfos)
		{
			int skip = connInfo.temporalSkip;
			if (frame >= skip)
				connInfo.conn->forward(frame - skip, frame);
			else
				connInfo.conn->prehistory_forward(
						prehistoryParams[connInfo.conn->inLayer],
						frame - skip, frame);
		}

		for (ComponentPtr compon : this->components)
			compon->forward(frame, frame);

		++ frame;
	}

	/**
	 * First back-props to the previous time point,
	 * then pass the gradient backward in current time.
	 */
	virtual void backward_prop()
	{
		-- frame;

		for (int i = components.size() - 1; i >= 0; --i)
			components[i]->backward(frame, frame);

		for (auto& connInfo : this->recurConnectionInfos)
		{
			int skip = connInfo.temporalSkip;
			if (frame >= skip)
				connInfo.conn->backward(frame, frame - skip);
			else
				connInfo.conn->prehistory_backward(
						prehistoryParams[connInfo.conn->inLayer],
						frame, frame - skip);
		}

		for (LayerPtr layer : layers)
			layer->shiftBackGradientWindow();
	}

	virtual void add_recurrent_connection(ConnectionPtr conn, int temporalSkip = 1)
	{
		assert_throw(
			maxTemporalSkip == Layer::UNLIMITED_TEMPORAL_SKIP
				|| temporalSkip <= maxTemporalSkip,
			NetworkException("temporalSkip should be <= maxTemporalSkip.\n"
					"Use set_max_temporal_skip() to change the upper limit.\n"
					"Then call assemble() again on the network"));

		recurConnectionInfos.push_back(RecurConnectionInfo(conn, temporalSkip));
		connections.push_back(conn);
		this->check_add_param_container(conn);

		// Add the largest temporal skip for the layer
		auto h_0 = prehistoryParams.find(conn->inLayer);
//		auto& dummyRand = FakeRand::instance(); // DUMMY
		auto& dummyRand = UniformFloatSingleton<-2, 4>::instance(); // DUMMY
		if (h_0 == prehistoryParams.end())
		{
			auto newh_0 = ParamContainer::make(temporalSkip);
			newh_0->fill_rand(dummyRand);
			prehistoryParams[conn->inLayer] = newh_0;
			paramContainers.push_back(newh_0);
		}
		else if (h_0->second->size() < temporalSkip)
		{
			h_0->second->resize(temporalSkip);
			h_0->second->fill_rand(dummyRand);
		}
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_recurrent_connection(ArgT&& ... args)
	{
		this->add_recurrent_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...));
	}

	template<typename ConnectionT, typename ...ArgT>
	void new_recurrent_skip_connection(int temporalSkip, ArgT&& ... args)
	{
		this->add_recurrent_connection(
			Connection::make<ConnectionT>(
					std::forward<ArgT>(args)...),
			temporalSkip);
	}

	virtual void reset()
	{
		for (ComponentPtr compon : this->components)
			compon->reset();
		for (auto& connInfo : this->recurConnectionInfos)
			connInfo.conn->reset();

		frame = 0;

		this->assemble();
	}

	/************************************/
	struct RecurConnectionInfo
	{
		RecurConnectionInfo(ConnectionPtr _conn, int _temporalSkip = 1) :
			conn(_conn), temporalSkip(_temporalSkip)
		{ }

		ConnectionPtr conn;
		int temporalSkip;
	};

	vector<RecurConnectionInfo> recurConnectionInfos;

	std::unordered_map<LayerPtr, ParamContainerPtr> prehistoryParams;

protected:
	int frame = 0;
	int maxTemporalSkip = 1;
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
	for (auto recConn : net.recurConnectionInfos)
		os << "  " << recConn.conn->str() << "\n";
	os << "]";
	return os;
}

#endif /* NETWORK_H_ */
