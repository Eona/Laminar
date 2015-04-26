/*
 * Eona Studio (c) 2015
 */

#ifndef UTILS_DEBUG_UTILS_H_
#define UTILS_DEBUG_UTILS_H_

#include <iostream>
#include <string>
#include <stdexcept>
#include <type_traits>

#undef assert
#define TERMINATE_ASSERT false
#define DEBUG true

/**
 * 'Press enter to continue...'
 */
void input_halt()
{
	do {
		std::cout << "Press Enter to continue ...";
	}
	while (std::cin.get() != '\n');
}

/**
 * static_assert doesn't support string concat
 * Use a macro to append a big title to the error message
 * WARNING if 'cond' has comman, like foo<int, int>, you must enclose it with parenthesis
 * (foo<int, int>) to avoid being interpreted as two separate macro args
 */
#define LMN_STATIC_ASSERT(cond, errmsg) \
	static_assert(cond, "\n\n\n\n\nLaminar static assert failure:\n" errmsg "\n\n\n\n\n")

/**
 * @param varname the generated errmsg will be "<varname> must be a subclass of <Baseclass>"
 */
#define LMN_STATIC_ASSERT_IS_BASE(Baseclass, Subclass, varname) \
	static_assert(std::is_base_of<Baseclass, Subclass>::value, \
		"\n\n\n\n\nLaminar static assert: incompatible inheritance\n" varname " must be a subclass of " #Baseclass "\n\n\n\n\n")

class AssertFailure: public std::exception {
protected:
    std::string msg;
public:
    AssertFailure(const std::string& _msg):
    	msg(_msg) {}

    virtual const char* what() const throw() {
        return (std::string("\n[Assert Error] ") + msg).c_str();
    }
};

inline void assert(bool cond, std::string errmsg = "", std::string successmsg="")
{
	if (!cond)
	{
	#if TERMINATE_ASSERT
		throw AssertFailure(errmsg);
	#else
		std::cout << "[Assert Error] " << errmsg << "\n";
	#endif
	}
	else if (successmsg != "")
		std::cout << successmsg << "\n";
}

/**
 * Macro equivalent to delay errmsg evaluation
 */
#define LMN_ASSERT_THROW(cond, exc) \
	{if (!(cond)) \
	{ \
		std::cerr << __FILE__ << "\n@ line " << __LINE__ \
				<< " Laminar assertion failure\n"; \
		throw exc; \
	}}

/**
 * Macro equivalent to delay errmsg evaluation
 */
#define LMN_ASSERT_NULLPTR(ptr, exc) \
	{if (ptr == nullptr) \
	{ \
		std::cerr << __FILE__ << "\n@ line " << __LINE__ \
				<< " Laminar assertion: should NOT be nullptr\n"; \
		throw exc; \
	}}

template<typename FloatT = float>
void assert_float_eq(FloatT f1, FloatT f2, FloatT tol = 1e-4f,
		std::string errmsg = "", std::string successmsg="")
{
	FloatT diff = fabs(f1 - f2);
	if (diff >= tol)
		errmsg += std::string(errmsg=="" ? "" : ":") +
			"\n(" + std::to_string(f1) + ") - (" +
			std::to_string(f2) + ") = " + std::to_string(diff);
	else if (successmsg != "")
		successmsg += std::string(": (") +
			std::to_string(f1) + ") == (" +
			std::to_string(f2) + ")";

	assert(diff < tol, errmsg, successmsg);
}

/**
 * If the difference percentage (w.r.t average of two operand values)
 * is greater than TOL, we assert failure.
 */
template<typename FloatT = float>
void assert_float_percent_eq(FloatT f1, FloatT f2, FloatT percentTol = 1.0f,
		std::string errmsg = "", std::string successmsg="")
{
	const FloatT DEFAULT_ABS_TOL = 1e-3f;
	if (fabs(f1) < DEFAULT_ABS_TOL || fabs(f2) < DEFAULT_ABS_TOL)
	{
		assert_float_eq(f1, f2, DEFAULT_ABS_TOL, errmsg, successmsg);
		return;
	}

	FloatT percentDiff = fabs((f1 - f2) / (0.5*(f1 + f2))) * 100;

	if (percentDiff >= percentTol)
		errmsg += std::string(errmsg=="" ? "" : ":") +
			"\n(" + std::to_string(f1) + ") != (" +
			std::to_string(f2) + ") -> " +
			std::to_string(percentDiff) + " % diff";
	else if (successmsg != "")
		successmsg += std::string(": (") +
			std::to_string(f1) + ") == (" +
			std::to_string(f2) + ")";

	assert(percentDiff < percentTol, errmsg, successmsg);
}

void print_title(std::string title = "", int leng = 10)
{
	std::string sep = "";
	for (int i = 0; i < leng; ++i)
		sep += "=";

	std::cout << sep << " " << title << " " << sep << " \n";
}

/**************************************
******* Laminar specific exceptions *********
**************************************/
class LaminarException: public std::exception
{
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

class NetworkException: public LaminarException
{
public:
    NetworkException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Network error";
    }
};

class ComponentException: public NetworkException
{
public:
    ComponentException(const std::string& msg):
    	NetworkException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Network component error";
    }
};

class EngineException: public LaminarException
{
public:
    EngineException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Engine error";
    }
};

class TensorException: public LaminarException
{
public:
    TensorException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Tensor error";
    }
};

class LearningException: public LaminarException
{
public:
    LearningException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Learning error";
    }
};

class DataException: public LaminarException
{
public:
    DataException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Data error";
    }
};


class UnimplementedException: public LaminarException
{
public:
    UnimplementedException(const std::string& msg):
    	LaminarException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "Feature unimplemented";
    }
};


/* *****
 *	Variadic MACRO utilities. Used mainly for debugging
 * Usage:
 * #define example(...) VARARG(example, __VA_ARGS__)
 * #define example_0() "with zero parameters"
 * #define example_1() "with 1 parameter"
 * #define example_3() "with 3 parameter"
 * // call as if the 'example' macro is overloaded
 * example() + example(66) + example(34, 23, 99)
 */
// The MSVC has a bug when parsing '__VA_ARGS__'. Workaround:
#define VA_EXPAND(x) x
// always return the fifth argument in place
#define VARARG_INDEX(_0, _1, _2, _3, _4, _5, ...) _5
// how many variadic parameters?
#define VARARG_COUNT(...) VA_EXPAND(VARARG_INDEX(__VA_ARGS__, 5, 4, 3, 2, 1))
#define VARARG_HELPER2(base, count, ...) base##_##count(__VA_ARGS__)
#define VARARG_HELPER(base, count, ...) VARARG_HELPER2(base, count, __VA_ARGS__)
#define VARARG(base, ...) VARARG_HELPER(base, VARARG_COUNT(__VA_ARGS__), __VA_ARGS__)
// Define DEBUG_MSG_1 or _2 or _n to define a debug message printout macro with n args
// intelliSense might underline this as syntax error. Ignore it and compile.
#define DEBUG_MSG(...) VARARG(DEBUG_MSG,	 __VA_ARGS__)

// More debugging info
#if DEBUG
#define DEBUG_MSG_1(msg) std::cout << msg << "\n"
#define DEBUG_MSG_2(name, msg) std::cout << "{" << name << "} " << msg << "\n"
#define DEBUG_TITLE(msg) print_title(msg)
#define DEBUG_DO(command) command
#define DEBUG_COND(cond, msg) if (cond) std::cout << msg << "\n"
#define DEBUG_LOOP(forcond, msg) for (forcond) std::cout << msg << "\n";
// Write to debug file
#define DEBUG_FILE_INIT(filename) ofstream fdbg(filename);
#define DEBUG_FOUT(msg) fdbg << msg << "\n"
#else
#define DEBUG_MSG_1(msg)
#define DEBUG_MSG_2(msg1, msg2)
#define DEBUG_TITLE(msg)
#define DEBUG_DO(command)
#define DEBUG_COND(cond, msg)
#define DEBUG_LOOP(forcond, msg)
#define DEBUG_FILE_INIT(filename)
#define DEBUG_FOUT(msg)
#endif


#endif /* UTILS_DEBUG_UTILS_H_ */
