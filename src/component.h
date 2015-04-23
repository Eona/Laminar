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
	Component() :
		initGuard("Component")
	{}

	virtual ~Component() {};

	void init_engine(EngineBase::Ptr engine)
	{
		initGuard.assert_before_initialize("init_engine");
		this->engine = engine;
	}

	virtual void initialize()
	{
		this->initialize_impl();

		// set *after* initialize_impl, which requires initGuard to be false
		initGuard.initialize();
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

	GEN_DOWN_CAST_STATIC_MEMBER(Component)

protected:
	EngineBase::Ptr engine;
	InitializeGuard<ComponentException> initGuard;

	virtual void initialize_impl() = 0;
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
