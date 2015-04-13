/*
 * Eona Studio (c) 2015
 */

#ifndef COMPONENT_H_
#define COMPONENT_H_

#include "global_utils.h"
#include "engine/engine.h"

class Component
{
public:
	Component() {}

	virtual ~Component() {};

	void init_engine(EngineBase::Ptr engine)
	{
		assert_throw(!this->is_initialized,
			ComponentException("init_engine() must be called before initialize()"));
		this->engine = engine;
	}

	virtual void initialize()
	{
		assert_throw(!this->is_initialized,
			ComponentException("already initialized, can't init again unless reset()"));

		this->initialize_impl();

		this->is_initialized = true;
	}

	virtual void reset()
	{
		this->reset_impl();
		this->is_initialized = false;
	}

	virtual void forward(int inFrame = 0, int outFrame = 0) = 0;

	virtual void backward(int outFrame = 0, int inFrame = 0) = 0;


	virtual explicit operator string() const = 0;

	/************************************/
	typedef shared_ptr<Component> Ptr;

	template<typename ComponentT>
	static Component::Ptr upcast(shared_ptr<ComponentT> compon)
	{
		return static_cast<Component::Ptr>(compon);
	}

	template<typename ComponentT>
	static shared_ptr<ComponentT> cast(Component::Ptr compon)
	{
		return std::dynamic_pointer_cast<ComponentT>(compon);
	}

protected:
	EngineBase::Ptr engine;
	bool is_initialized = false;

	virtual void initialize_impl() = 0;

	virtual void reset_impl() = 0;

	/**
	 * Utility: construct a Tensor, null created or with dim
	 */
	Tensor create_tensor(vector<int> dim = {})
	{
		if (dim.empty())
			return Tensor(engine);
		else
			return Tensor(engine, dim);
	}
};

TYPEDEF_PTR(Component);

template<typename T>
typename enable_if<is_base_of<Component, T>::value, ostream>::type&
operator<<(ostream& os, T& compon)
{
	os << compon.str();
	return os;
}

template<typename T>
typename enable_if<is_base_of<Component, T>::value, ostream>::type&
operator<<(ostream& os, T&& compon)
{
	os << compon.str();
	return os;
}

#endif /* COMPONENT_H_ */
