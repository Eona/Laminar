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

	virtual void forward(int inTime = 0, int outTime = 0) = 0;

	virtual void backward(int outTime = 0, int inTime = 0) = 0;

	virtual void reset() = 0;

	virtual string str() = 0;

protected:
	// utility: grow vector on demand
	static inline void resize_on_demand(vector<float>& vec, int accessIdx)
	{
		if (accessIdx >= vec.size())
			vec.resize(accessIdx + 1, 0);
	}
};

TypedefPtr(Component);

template<typename ComponentT>
ComponentPtr make_component(shared_ptr<ComponentT> compon)
{
	return static_cast<ComponentPtr>(compon);
}

template<typename ComponentT>
shared_ptr<ComponentT> cast_component(ComponentPtr compon)
{
	return std::dynamic_pointer_cast<ComponentT>(compon);
}

#endif /* COMPONENT_H_ */
