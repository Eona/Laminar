/*
 * Eona Studio (c) 2015
 */

#ifndef LAMINAR_UTILS_H_
#define LAMINAR_UTILS_H_

#include "global_utils.h"

/**
 * Inherit from this tagging interface to work with gradient check
 */
template<typename FloatT>
class GradientCheckable
{
public:
	virtual ~GradientCheckable() {}

	// restore() calls must correspond one-by-one to perturb() calls
	void gradient_check_perturb(int changeItem, DimIndex dimIdx, FloatT eps)
	{
		lastChangeItem = changeItem;
		lastDimIdx = dimIdx;
		lastEps = eps;
		this->gradient_check_perturb_impl(changeItem, dimIdx, eps);
	}

	void gradient_check_restore()
	{
		this->gradient_check_restore_impl(lastChangeItem, lastDimIdx, lastEps);
	}

protected:
	/**
	 * Subclass needs to implement the following
	 */
	virtual void gradient_check_perturb_impl(
			int changeItem, DimIndex dimIdx, FloatT eps) = 0;

	virtual void gradient_check_restore_impl(
			int lastChangeItem, DimIndex lastDimIdx, FloatT lastEps) = 0;

private:
	// For restoring
	int lastChangeItem;
	DimIndex lastDimIdx;
	FloatT lastEps;
};

/**
 * Check if a given class is a subclass of GradientCheckable<FloatT> for any FloatT
 */
GEN_IS_DERIVED_TEMPLATE_TRAIT(is_gradient_checkable, GradientCheckable);

#endif /* LAMINAR_UTILS_H_ */
