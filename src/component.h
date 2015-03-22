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

	virtual ~Component() {}

	virtual void forward() = 0;

	virtual void backward() = 0;

	virtual string str() = 0;
};

typedef shared_ptr<Component> ComponentPtr;

template<typename ComponentT>
ComponentPtr makeComponent(shared_ptr<ComponentT> ptr)
{
	return static_cast<ComponentPtr>(ptr);
}

#endif /* COMPONENT_H_ */
