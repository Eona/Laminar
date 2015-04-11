/*
 * Eona Studio (c) 2015
 */

#ifndef COMPONENT_H_
#define COMPONENT_H_

#include "global_utils.h"

class Component
{
public:
	Component() {}

	virtual ~Component() {};

	virtual void forward(int inFrame = 0, int outFrame = 0) = 0;

	virtual void backward(int outFrame = 0, int inFrame = 0) = 0;

	virtual void reset() = 0;

	virtual string str() = 0;

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
};

TypedefPtr(Component);

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
