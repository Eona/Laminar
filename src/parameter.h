/*
 * Eona Studio (c) 2015
 */


#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "global_utils.h"

class ParameterContainer
{
public:
	ParameterContainer(int numberOfParam = 1) :
		paramValues(numberOfParam),
		paramGradients(numberOfParam)
	{ }

	~ParameterContainer() =default;

	void reset()
	{
		for (int i = 0; i < paramValues.size(); ++i)
		{
			paramValues[i] = 0;
			paramGradients[i] = 0;
		}
	}

	vector<float> paramValues;
	vector<float> paramGradients;
};

#endif /* PARAMETER_H_ */
