/*
 * myutils.h
 * (c) 2015
 * Author: Jim Fan
 * Common C++ header inclusion and print/vector utils
 */
#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <memory>
#include <fstream>
#include <vector>
#include <stack>
#include <queue>
#include <deque>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <climits>
#include <cstdarg>
#include <utility>
#include <type_traits>

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::ostringstream;
using std::move;
using std::shared_ptr;
using std::function;
using std::enable_if;
using std::is_base_of;
typedef unsigned long ulong;
typedef unsigned int uint;

#if __cplusplus > 201100l
#define is_CPP_11
#else
#undef is_CPP_11
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
#define is_CUDA
#else
#undef is_CUDA
#endif

// anonymous namespace to avoid multiple definition linker error
namespace {
/**************************************
************ Printing **************
**************************************/
template<typename Container>
string container2str(Container&& vec,
		string leftDelimiter="[", string rightDelimiter="]")
{
	std::ostringstream oss;
	oss << leftDelimiter;
	for (int i = 0; i < vec.size(); ++i)
		oss << vec[i] << ", ";
	string s = oss.str();
	return (s.size() > leftDelimiter.size() ?
			s.substr(0, s.size() - 2) : s) + rightDelimiter;
}

template<typename T>
ostream& operator<<(ostream& oss, const vector<T>& vec)
{
	return oss << container2str(vec);
}
template<typename T>
ostream& operator<<(ostream& oss, vector<T>&& vec)
{
	return oss << vec;
}

template<typename T>
inline string to_str(T val)
{
	ostringstream oss;
	oss << val;
	return oss.str();
}

// print basic array
template <typename T>
void print_array(T *arr, int size)
{
	cout << "[";
	int i;
	for (i = 0; i < size - 1; ++i)
		cout << arr[i] << ", ";
	cout << arr[i] << "]\n";
}

inline bool starts_with(string str, string prefix)
{
	return str.find(prefix) == 0;
}

/**************************************
************ Misc **************
**************************************/
// Define shared_ptr<Xclass> as XclassPtr
#define TYPEDEF_PTR(Xclass) \
	typedef shared_ptr<Xclass> Xclass##Ptr

// Emulate python style subscript
template<typename T>
inline T& vec_at(vector<T>& vec, int idx)
{
	if (idx < 0)
		idx += (int) vec.size();
	return vec[idx];
}

// utility: grow vector on demand
template<typename T>
inline void vec_resize_on_demand(vector<T>& vec, int accessIdx)
{
// WARNING when comparing negative number with size_t, -1 will be converted to positive!!!
	if (accessIdx >= (int) vec.size())
		vec.resize(accessIdx + 1, 0);
}

template<typename T, typename Func>
inline void vec_apply(vector<T>& vec, Func& f)
{
	for (T& elem : vec)
		elem = f();
}

template <typename KeyT, typename ValueT>
bool key_exists(std::unordered_map<KeyT, ValueT>& map, KeyT& key)
{
	return map.find(key) != map.end();
}

// enclose x as "x" in macro expansion
#define _STRINGFY(x) #x
#define STRINGFY(x) _STRINGFY(x)

/**
 * Example: suppose we have template<typename T> MyClass;
 * GenDerivedTemplateTypeTrait(is_myclass, MyClass)
 * generates a type trait function is_myclass<X>() that
 * returns true if there exists a type T such that X inherits from MyClass<T>
 *
 * std::declval() simulates creation of a new Derived* pointer
 * http://stackoverflow.com/questions/29531536/c-check-inheritance-at-compile-time/
 */
#define GEN_IS_DERIVED_TEMPLATE_TRAIT(typeTraitFuncName, className) \
template <typename T> \
std::true_type typeTraitFuncName##_impl(const className<T>* impl); \
std::false_type typeTraitFuncName_impl(...); \
template <typename Derived> \
using typeTraitFuncName = \
    decltype(typeTraitFuncName##_impl(std::declval<Derived*>()));

/**
 * Example: suppose we have template<typename T> MyClass;
 * GenTemplateTypeTrait(is_myclass, MyClass)
 * generates a type trait function is_myclass<X>() that
 * returns true if there exists a type T such that X == MyClass<T>
 */
#define GEN_IS_TEMPLATE_TRAIT(typeTraitFuncName, className) \
template <typename T> \
struct typeTraitFuncName : std::false_type { }; \
template <typename... Ts> \
struct typeTraitFuncName<className<Ts...> > : std::true_type { };

GEN_IS_TEMPLATE_TRAIT(is_vector, std::vector);

/**
 * select_type<bool, TypeTrue, TypeFalse>::type
 * if bool is true, select TypeTrue, else TypeFalse
 */
template<bool, typename TypeTrue, typename TypeFalse>
struct select_type
{
	using type = TypeFalse;
};
template<typename TypeTrue, typename TypeFalse>
struct select_type<true, TypeTrue, TypeFalse>
{
	using type = TypeTrue;
};


/**************************************
************ Exceptions **************
**************************************/
class AssertFailure: public std::exception {
protected:
    std::string msg;
public:
    AssertFailure(const string& _msg):
    	msg(_msg) {}

    virtual const char* what() const throw() {
        return (string("\n[Assert Error] ") + msg).c_str();
    }
};

class LaminarException: public std::exception {
protected:
    std::string msg;
public:
    LaminarException(const string& _msg):
    	msg(_msg) {}

    // all sub-exceptions need to override this
    virtual string error_header() const
    {
    	return "General error";
    }

    virtual const char* what() const throw()
	{
        return (string("[") + error_header() + "] " + msg).c_str();
    }
};

class NetworkException: public LaminarException {
public:
    NetworkException(const string& msg):
    	LaminarException(msg)
	{}

    virtual string error_header() const
    {
    	return "Network error";
    }
};

class ComponentException: public LaminarException {
public:
    ComponentException(const string& msg):
    	LaminarException(msg)
	{}

    virtual string error_header() const
    {
    	return "Network component error";
    }
};

class UnimplementedException: public LaminarException {
public:
    UnimplementedException(const string& msg):
    	LaminarException(msg)
	{}

    virtual string error_header() const
    {
    	return "Feature unimplemented";
    }
};

class EngineException: public LaminarException {
public:
    EngineException(const string& msg):
    	LaminarException(msg)
	{}

    virtual string error_header() const
    {
    	return "Engine error";
    }
};

class TensorException: public LaminarException {
public:
    TensorException(const string& msg):
    	LaminarException(msg)
	{}

    virtual string error_header() const
    {
    	return "Tensor error";
    }
};

/**************************************
************ Debugging **************
**************************************/
#undef assert
#define TERMINATE_ASSERT false
#define DEBUG true

inline void assert(bool cond, string errmsg = "", string successmsg="")
{
	if (!cond)
	{
	#if TERMINATE_ASSERT
		throw AssertFailure(errmsg);
	#else
		cout << "[Assert Error] " << errmsg << "\n";
	#endif
	}
	else if (successmsg != "")
		cout << successmsg << "\n";
}

template<typename T>
typename enable_if<is_base_of<std::exception, T>::value, void>::type
assert_throw(bool cond, T&& throwable)
{
	if (!cond)
		throw throwable;
}

template<typename FloatT>
void assert_float_eq(FloatT f1, FloatT f2, FloatT tol = 1e-4f,
		string errmsg = "", string successmsg="")
{
	FloatT diff = abs(f1 - f2);
	if (diff >= tol)
		errmsg += string(errmsg=="" ? "" : ":") +
			"\n(" + to_str(f1) + ") - (" +
			to_str(f2) + ") = " + to_str(diff);
	else if (successmsg != "")
		successmsg += string(": (") +
			to_str(f1) + ") == (" +
			to_str(f2) + ")";

	assert(diff < tol, errmsg, successmsg);
}

/**
 * If the difference percentage (w.r.t average of two operand values)
 * is greater than TOL, we assert failure.
 */
template<typename FloatT>
void assert_float_percent_eq(FloatT f1, FloatT f2, FloatT percentTol = 1.0f,
		string errmsg = "", string successmsg="")
{
	const float DEFAULT_ABS_TOL = 1e-3f;
	if (abs(f1) < DEFAULT_ABS_TOL || abs(f2) < DEFAULT_ABS_TOL)
	{
		assert_float_eq(f1, f2, DEFAULT_ABS_TOL, errmsg, successmsg);
		return;
	}

	FloatT percentDiff = abs((f1 - f2) / (0.5*(f1 + f2))) * 100;

	if (percentDiff >= percentTol)
		errmsg += string(errmsg=="" ? "" : ":") +
			"\n(" + to_str(f1) + ") != (" +
			to_str(f2) + ") -> " +
			to_str(percentDiff) + " % diff";
	else if (successmsg != "")
		successmsg += string(": (") +
			to_str(f1) + ") == (" +
			to_str(f2) + ")";

	assert(percentDiff < percentTol, errmsg, successmsg);
}

void print_title(string title = "", int leng = 10)
{
	string sep = "";
	for (int i = 0; i < leng; ++i)
		sep += "=";

	cout << sep << " " << title << " " << sep << " \n";
}

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
#define DEBUG_MSG_1(msg) cout << msg << "\n"
#define DEBUG_MSG_2(name, msg) cout << "{" << name << "} " << msg << "\n"
#define DEBUG_TITLE(msg) print_title(msg)
#define DEBUG_DO(command) command
#define DEBUG_COND(cond, msg) if (cond) cout << msg << "\n"
#define DEBUG_LOOP(forcond, msg) for (forcond) cout << msg << "\n";
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

} // end of anonymous namespace
#endif /* UTILS_H_ */
