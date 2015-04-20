/*
 * Eona Studio (c) 2015
 */

#ifndef COMPONENT_H_
#define COMPONENT_H_

#include "engine/engine.h"
#include "utils/debug_utils.h"
#include "utils/global_utils.h"
#include "utils/laminar_utils.h"

class Component
{
public:
	Component() {}

	virtual ~Component() {};

	void init_engine(EngineBase::Ptr engine)
	{
		check_uninitialized("init_engine", "Component");
		this->engine = engine;
	}

	virtual void initialize()
	{
		assert_throw<ComponentException>(!this->is_initialized,
			"Component already initialized, can't init again unless reset()");

		this->initialize_impl();

		this->is_initialized = true;
	}

	/**
	 * Clear all values, gradients and weight parameters to 0
	 */
	virtual void zero_clear() = 0;

	// TODO
//	virtual void reset() = 0;

	virtual void forward(int inFrame = 0, int outFrame = 0) = 0;

	virtual void backward(int outFrame = 0, int inFrame = 0) = 0;

	virtual explicit operator string() const = 0;

	/************************************/
	TYPEDEF_PTR(Component);

	template<typename ComponentT>
	static Component::Ptr upcast(std::shared_ptr<ComponentT> compon)
	{
		return std::static_pointer_cast<Component>(compon);
	}

	template<typename ComponentT>
	static std::shared_ptr<ComponentT> cast(Component::Ptr compon)
	{
		LAMINAR_STATIC_ASSERT((std::is_base_of<Component, ComponentT>::value),
				"cast() failed: type parameter must be a subclass of Component");
		return std::dynamic_pointer_cast<ComponentT>(compon);
	}

protected:
	EngineBase::Ptr engine;
	bool is_initialized = false;

	virtual void initialize_impl() = 0;

	/**
	 * Exception helper
	 * The function should be called *before* initialization
	 */
	void check_uninitialized(string msg, string componentName)
	{
		assert_throw<ComponentException>(!this->is_initialized,
			msg + " should be called before " + componentName + " initialization.");
	}
};

TYPEDEF_PTR_EXTERNAL(Component);

template<typename T>
typename std::enable_if<std::is_base_of<Component, T>::value, std::ostream>::type&
operator<<(std::ostream& os, T& compon)
{
	os << compon.str();
	return os;
}

template<typename T>
typename std::enable_if<std::is_base_of<Component, T>::value, std::ostream>::type&
operator<<(std::ostream& os, T&& compon)
{
	os << compon.str();
	return os;
}

#endif /* COMPONENT_H_ */
