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

/*********** A few typedefs ***********/
/**
 * Define shared_ptr<Xclass> as ::Ptr
 * Use inside a class definition
 */
#define TYPEDEF_PTR(Xclass) \
	typedef std::shared_ptr<Xclass> Ptr

/**
 * Define shared_ptr<Xclass> as XclassPtr
 * Use outside a class definition
 */
#define TYPEDEF_PTR_EXTERNAL(Xclass) \
	typedef std::shared_ptr<Xclass> Xclass##Ptr

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

private:
	Dimension totalDim;
	DimIndex current; //current idx
	bool hasNext;
	DimIndex last;
};


/**************************************
******* Laminar specific exceptions *********
**************************************/
class LaminarException: public std::exception {
protected:
    std::string msg;
public:
    LaminarException(const std::string& _msg):
    	msg(_msg) {}

    // all sub-exceptions need to override this
    virtual std::string error_header() const
    {
    	return "General error";
    }

    virtual const char* what() const throw()
	{
        return (std::string("[") + error_header() + "] " + msg).c_str();
    }
};

class NetworkException: public LaminarException {
public:
    NetworkException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Network error";
    }
};

class ComponentException: public LaminarException {
public:
    ComponentException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Network component error";
    }
};

class UnimplementedException: public LaminarException {
public:
    UnimplementedException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Feature unimplemented";
    }
};

class EngineException: public LaminarException {
public:
    EngineException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Engine error";
    }
};

class TensorException: public LaminarException {
public:
    TensorException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Tensor error";
    }
};

#endif /* UTILS_LAMINAR_UTILS_H_ */
