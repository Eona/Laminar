/*
 * Eona Studio (c) 2015
 */


#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "global_utils.h"
#include "engine/tensor_ops.h"

class ParamContainer
{
public:
	ParamContainer(int size = 1) :
		paramValues(size),
		paramGradients(size)
	{ }

	virtual ~ParamContainer() {};

	void clear_values()
	{
		for (auto ptr : paramValues)
			lmn::clear(*ptr);
	}

	void clear_gradients()
	{
		for (auto ptr : paramGradients)
			lmn::clear(*ptr);
	}

	/**
	 * Holders of ParamContainer are responsible for initializing the tensors
	 * Assign to the returned ref to initialize the Tensor::Ptr
	 * @return ref to Tensor::Ptr
	 */
	Tensor::Ptr& get_param_value(int idx)
	{
		return this->paramValues[idx];
	}

	Tensor::Ptr& get_param_gradient(int idx)
	{
		return this->paramGradients[idx];
	}

	vector<Tensor::Ptr>& param_values()
	{
		return this->paramValues;
	}

	vector<Tensor::Ptr>& param_gradients()
	{
		return this->paramGradients;
	}

	int size() const
	{
		return paramValues.size();
	}

	/************************************/
	TYPEDEF_PTR(ParamContainer);

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
	// TODO add 2D index
	void gradient_check_perturb(int changeIdx, float eps)
	{
		lastChangedIdx = changeIdx;
		lastEps = eps;
		lmn::perturb(*paramValues[changeIdx], {}, eps);
	}

	void gradient_check_restore()
	{
		lmn::perturb(*paramValues[lastChangedIdx], {}, -lastEps);
	}

	/************************************/
private:
	vector<Tensor::Ptr> paramValues;
	vector<Tensor::Ptr> paramGradients;

	int lastChangedIdx; float lastEps; // DEBUG ONLY
};

TYPEDEF_PTR_EXTERNAL(ParamContainer);

#endif /* PARAMETER_H_ */
