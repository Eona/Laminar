/*
 * Eona Studio (c) 2015
 */


#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "global_utils.h"

class ParamContainer
{
public:
	ParamContainer(int numberOfParam = 1) :
		paramValues(numberOfParam),
		paramGradients(numberOfParam)
	{ }

	~ParamContainer() =default;

	void reset()
	{
		for (int i = 0; i < paramValues.size(); ++i)
		{
			paramValues[i] = 0;
			paramGradients[i] = 0;
		}
	}

	/************************************/
	typedef shared_ptr<ParamContainer> Ptr;

	template<typename ParamContainerT>
	static ParamContainer::Ptr upcast(shared_ptr<ParamContainerT> compon)
	{
		return std::dynamic_pointer_cast<ParamContainer>(compon);
	}

	template<typename ParamContainerT>
	static shared_ptr<ParamContainerT> cast(ParamContainer::Ptr param)
	{
		return std::dynamic_pointer_cast<ParamContainerT>(param);
	}

	/*********** DEBUG ONLY ***********/
	int lastChangedIdx;
	float oldValue;

	// restore() calls must correspond one-by-one to perturb() calls
	void gradient_check_perturb(int changeIdx, float eps)
	{
		lastChangedIdx = changeIdx;
		oldValue = paramValues[changeIdx];
		paramValues[changeIdx] += eps;
	}

	void gradient_check_restore()
	{
		paramValues[lastChangedIdx] = oldValue;
	}

	/************************************/
	vector<float> paramValues;
	vector<float> paramGradients;
};

TypedefPtr(ParamContainer);

#endif /* PARAMETER_H_ */
