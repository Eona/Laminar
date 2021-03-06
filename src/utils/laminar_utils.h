/*
 * Eona Studio (c) 2015
 */

#ifndef UTILS_LAMINAR_UTILS_H_
#define UTILS_LAMINAR_UTILS_H_

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "debug_utils.h"

/*********** A few typedefs ***********/
/**
 * Type alias for vector<int>
 */
typedef std::vector<int> Dimension;
typedef std::vector<int> DimIndex;

/**
 * Inherit from this tagging interface to work with gradient check
 */
template<typename FloatT = float>
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
	 * Derived classes need to implement the following
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
 * Enumerate dimension indices.
 * e.g. for a 3*2 tensor, the sequence generated will be:
 * (0,0); (1,0); (2,0); (0,1); (1,1); (2,1)
 */
class DimIndexEnumerator
{
public:
	DimIndexEnumerator(Dimension totalDim_) :
		totalDim(totalDim_),
		// start at all zeros
		current(totalDim.size(), 0)
	{
		hasNext = !totalDim.empty();
	}

	bool has_next() const
	{
		return this->hasNext;
	}

	/**
	 * @return next index, increment from the leftmost dimension
	 */
	DimIndex next()
	{
		// To start with [0, 0, 0]
		last = current;
		for (int di = 0; di < totalDim.size(); ++di)
		{
			if (current[di] == totalDim[di] - 1)
			{
				current[di] = 0;
				if (di + 1 == totalDim.size())
				{
					this->hasNext = false;
					return last; // no next DimIndex, enumeration completed
				}
			}
			else
			{
				current[di] += 1;
				break;
			}
		}
		return last;
	}

	/**
	 * Calculate linearized position of this index
	 * Consistent with next() generation order
	 * For example, if total dimension is [8, 9, 10], a DimIndex [4, 3, 2]
	 * is linearized to 4 + (8) * 3 + (8*9) * 2
	 */
	int linearize(DimIndex idx)
	{
		int baseSize = 1; // running product
		int ans = 0;
		for (int di = 0; di < totalDim.size(); ++di)
		{
			ans += baseSize * idx[di];
			baseSize *= this->totalDim[di];
		}
		return ans;
	}

private:
	Dimension totalDim;
	DimIndex current; //current idx
	bool hasNext;
	DimIndex last;
};

/**
 * Initialization guard:
 * Some classes should be initialized only once, unless explicitly reset
 * some class methods should be called before initialization, while
 * others should be called after.
 */
template<typename InitGuardExceptionT>
class InitializeGuard
{
public:
	InitializeGuard(string className_) :
		className(className_)
	{}

	~InitializeGuard() {}

	/**
	 * Use in initialize() method of your class
	 *
	 * @param disallowReinitialize if true,
	 * we will check if initialize() is called twice
	 */
	template<typename ExceptionT = InitGuardExceptionT>
	void initialize(bool disallowReinitialize = true)
	{
		if (disallowReinitialize)
			LMN_ASSERT_THROW(!isInited,
				ExceptionT(className + " should not be reinitialized."));

		this->isInited = true;
	}

	void reset()
	{
		this->isInited = false;
	}

	/**
	 * Use in a method that must be called before initialization
	 * Throw an exception if already initialized
	 * @param methodName for error message
	 * @param className to override the className given in ctor
	 */
	template<typename ExceptionT = InitGuardExceptionT>
	void assert_before_initialize(string methodName, string className) const
	{
		LMN_STATIC_ASSERT_IS_BASE(std::exception, ExceptionT,
						"assert_before_initialize template arg");

		LMN_ASSERT_THROW(!isInited,
			ExceptionT(methodName + " must be called before " + className + " initialization."));
	}

	/**
	 * @param methodName
	 * @param className use the one provided in ctor
	 */
	template<typename ExceptionT = InitGuardExceptionT>
	void assert_before_initialize(string methodName) const
	{
		assert_before_initialize<ExceptionT>(methodName, this->className);
	}

	/**
	 * Use in a method that must be called after initialization
	 * Throw an exception if not yet initialized
	 * @param methodName for error message
	 * @param className to override the className given in ctor
	 */
	template<typename ExceptionT = InitGuardExceptionT>
	void assert_after_initialize(string methodName, string className) const
	{
		LMN_STATIC_ASSERT_IS_BASE(std::exception, ExceptionT,
						"assert_after_initialize template arg");

		LMN_ASSERT_THROW(isInited,
			ExceptionT(methodName + " must be called after " + className + " initialization."));
	}

	/**
	 * @param methodName
	 * @param className use the one provided in ctor
	 */
	template<typename ExceptionT = InitGuardExceptionT>
	void assert_after_initialize(string methodName) const
	{
		assert_after_initialize<ExceptionT>(methodName, this->className);
	}

private:
	bool isInited = false;
	string className;
};


/**
 * Monitors change from last call
 */
template<typename ReturnT, typename ... ArgT>
struct ChangeMonitor
{
	typedef std::function<ReturnT(ArgT...)> FuncT;

	ChangeMonitor(FuncT func) :
		func(func)
	{}

	/**
	 * @return true if something has changed since the last monitor
	 * NOTE we use a different ArgT_ due to forwarding issues
	 */
	template<typename ... ArgT_>
	bool monitor(ArgT_&& ... args)
	{
		ReturnT now = func(std::forward<ArgT_>(args) ...);

		if (isFirst)
		{
			isFirst = false;
			cur = now;
			prev = now;
			return false;
		}

		bool isChanged = cur != now;
		prev = cur;
		cur = now;
		return isChanged;
	}

	ReturnT current() { return this->cur; }

	ReturnT previous() { return this->prev; }

private:
	FuncT func;
	ReturnT cur;
	ReturnT prev;
	bool isFirst = true;
};


/**************************************
******* Misc *********
**************************************/
/**
 * Generate static downcast member method for an abstract super class
 * The signature will be:
 * shared_ptr<Derived> Super::cast<Derived>(shared_ptr<Base> ptr)
 * will return nullptr if the cast fails
 */
#define GEN_DOWN_CAST_STATIC_MEMBER(Base) \
template<typename Derived> \
static std::shared_ptr<Derived> cast(std::shared_ptr<Base> superPtr) \
{ \
	LMN_STATIC_ASSERT_IS_BASE(Base, Derived, "cast() failure: type parameter"); \
	return std::dynamic_pointer_cast<Derived>(superPtr); \
	/* \
	auto subPtr = std::dynamic_pointer_cast<Derived>(superPtr); \
	LMN_ASSERT_NULLPTR(subPtr, \
		ExceptionType(#Base " down cast failure")); \
	return subPtr; \
	*/ \
}

/**
 * Generate static 'make' member method for an abstract base class
 * The signature will be:
 * shared_ptr<Derived> Base::make<Derived>(ArgT ... args)
 */
#define GEN_GENERIC_MAKEPTR_STATIC_MEMBER(Base) \
template<typename Derived, typename ...ArgT> \
static std::shared_ptr<Derived> make(ArgT&& ... args) \
{ \
	LMN_STATIC_ASSERT_IS_BASE(Base, Derived, "make() failure: type parameter"); \
	return std::make_shared<Derived>(std::forward<ArgT>(args) ...); \
}

/**
 * Generate static 'make' member method for a concrete class
 * The signature will be:
 * shared_ptr<Class> Class::make(ArgT ... args)
 * simply a shared_ptr version of the constructor, doesn't do any casting
 */
#define GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(Myclass) \
template<typename ...ArgT> \
static std::shared_ptr<Myclass> make(ArgT&& ... args) \
{ \
	return std::make_shared<Myclass>(std::forward<ArgT>(args) ...); \
}

#endif /* UTILS_LAMINAR_UTILS_H_ */
