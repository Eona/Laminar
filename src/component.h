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


#endif /* COMPONENT_H_ */
