/*
 * Eona Studio (c) 2015
 */


#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "global_utils.h"

class ParamContainer
{
public:
	ParamContainer(int size = 1) :
		paramValues(size),
		paramGradients(size)
	{ }

	~ParamContainer() =default;

	void resetValues()
	{
		std::fill(paramValues.begin(), paramValues.end(), 0);
	}

	void resetGradients()
	{
		std::fill(paramGradients.begin(), paramGradients.end(), 0);
	}

	int size()
	{
		return paramValues.size();
	}

	void resize(int newSize)
	{
		paramValues.resize(newSize);
		paramGradients.resize(newSize);
	}

	template<typename RandEngineT>
	void fill_rand(RandEngineT& randEngine)
	{
		for (int i = 0; i < size(); ++i)
			paramValues[i] = randEngine();
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

	template<typename ...ArgT>
	static ParamContainer::Ptr make(ArgT&& ... args)
	{
		return std::make_shared<ParamContainer>(
						std::forward<ArgT>(args) ...);
	}

	/*********** DEBUG ONLY ***********/
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

private:
	int lastChangedIdx; float oldValue; // DEBUG ONLY
};

TypedefPtr(ParamContainer);

#endif /* PARAMETER_H_ */
