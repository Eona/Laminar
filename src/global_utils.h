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
typedef unsigned long ulong;
typedef unsigned int uint;

/****** Recognition macros ******/
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
string container2str(Container& vec,
		string leftDelimiter="[", string rightDelimiter="]")
{
//	using ElemType = typename Container::value_type;
	std::ostringstream oss;
	oss << leftDelimiter;
//	for (ElemType& ele : vec)
//		oss << ele << ", ";
	for (int i = 0; i < vec.size(); ++i)
		oss << vec[i] << ", ";
	string s = oss.str();
	return (s.size() > leftDelimiter.size() ?
			s.substr(0, s.size() - 2) : s) + rightDelimiter;
}

template<typename T>
ostream& operator<<(ostream& oss, vector<T>& vec)
{
	return oss << container2str(vec);
}

template<typename T>
string to_str(T val)
{
	ostringstream oss;
	oss << val;
	return oss.str();
}

/****** Rvalue overloaded printing ******/
#ifdef is_CPP_11
template<typename Container>
string container2str(Container&& vec,
		string leftDelimiter="[", string rightDelimiter="]")
{
	return container2str(vec, leftDelimiter, rightDelimiter);
}

template<typename T>
ostream& operator<<(ostream& oss, vector<T>&& vec)
{
	return oss << vec;
}
#endif

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

/**************************************
************ Misc **************
**************************************/
// Define shared_ptr<Xclass> as XclassPtr
#define TypedefPtr(Xclass) \
	typedef shared_ptr<Xclass> Xclass##Ptr

// Type trait
template <typename Container>
struct is_vector : std::false_type { };
template <typename... Ts>
struct is_vector<std::vector<Ts...> > : std::true_type { };

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

/**************************************
************ Debugging **************
**************************************/
#undef assert
#define TERMINATE_ASSERT true
#define DEBUG true

void assert(bool cond, string errmsg = "", string successmsg="")
{
	if (!cond)
	{
	#if TERMINATE_ASSERT
		throw AssertFailure(errmsg);
	#else
		cerr << "[Assert Error] " << errmsg << endl;
	#endif
	}
	else if (successmsg != "")
		cout << successmsg << endl;
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
#define DEBUG_MSG_1(msg) cout << msg << endl
#define DEBUG_MSG_2(name, msg) cout << "{" << name << "} " << msg << endl
#define DEBUG_DO(command) command
#define DEBUG_COND(cond, msg) if (cond) cout << msg << endl
#define DEBUG_LOOP(forcond, msg) for (forcond) cout << msg << endl;
// Write to debug file
#define DEBUG_FILE_INIT(filename) ofstream fdbg(filename);
#define DEBUG_FOUT(msg) fdbg << msg << endl
#else
#define DEBUG_MSG_1(msg)
#define DEBUG_MSG_2(msg1, msg2)
#define DEBUG_DO(command)
#define DEBUG_COND(cond, msg)
#define DEBUG_LOOP(forcond, msg)
#define DEBUG_FILE_INIT(filename)
#define DEBUG_FOUT(msg)
#endif

} // end of anonymous namespace
#endif /* UTILS_H_ */
