/*
 * Eona Studio (c) 2015
 */


#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "engine/tensor_ops.h"
#include "utils/global_utils.h"
#include "utils/laminar_utils.h"

class ParamContainer : public GradientCheckable<float>
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
			lmn::zero_clear(*ptr);
	}

	void clear_gradients()
	{
		for (auto ptr : paramGradients)
			lmn::zero_clear(*ptr);
	}

	/**
	 * Holders of ParamContainer are responsible for initializing the tensors
	 * Assign to the returned ref to initialize the Tensor::Ptr
	 * @return ref to Tensor::Ptr
	 */
	Tensor::Ptr& param_value_ptr(int idx)
	{
		return this->paramValues[idx];
	}

	Tensor::Ptr& param_gradient_ptr(int idx)
	{
		return this->paramGradients[idx];
	}

	int size() const
	{
		return paramValues.size();
	}

	/************************************/
	TYPEDEF_PTR(ParamContainer);

	template<typename ParamContainerT>
	static ParamContainer::Ptr upcast(std::shared_ptr<ParamContainerT> compon)
	{
		return std::dynamic_pointer_cast<ParamContainer>(compon);
	}

	template<typename ParamContainerT>
	static std::shared_ptr<ParamContainerT> cast(ParamContainer::Ptr param)
	{
		return std::dynamic_pointer_cast<ParamContainerT>(param);
	}

	template<typename ...ArgT>
	static ParamContainer::Ptr make(ArgT&& ... args)
	{
		return std::make_shared<ParamContainer>(
						std::forward<ArgT>(args) ...);
	}

	/*********** Gradient checking ***********/
	/**
	 * GradientCheckable<float> interface
	 */
	virtual void gradient_check_perturb_impl(
			int changeItem, DimIndex dimIdx, float eps)
	{
		lmn::perturb(*paramValues[changeItem], dimIdx, eps);
	}

	/**
	 * GradientCheckable<float> interface
	 */
	virtual void gradient_check_restore_impl(
			int lastChangeItem, DimIndex lastDimIdx, float lastEps)
	{
		lmn::perturb(*paramValues[lastChangeItem], lastDimIdx, -lastEps);

	}

private:
	vector<Tensor::Ptr> paramValues;
	vector<Tensor::Ptr> paramGradients;
};

TYPEDEF_PTR_EXTERNAL(ParamContainer);

#endif /* PARAMETER_H_ */
