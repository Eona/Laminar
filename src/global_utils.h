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

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::ostringstream;
using std::move;
using std::shared_ptr;
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

/**************************************
************ Debugging **************
**************************************/
#undef assert

void assert(bool cond, string errmsg = "", string successmsg="")
{
	if (!cond)
	{
		cerr << "[Assert Fail] " <<  errmsg << endl;
		exit(1);
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
		errmsg = string(errmsg=="" ? "" : ":") +
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

/****** Exceptions ******/
class NeuralException: public std::exception {
private:
    std::string msg;
public:
    NeuralException(const string& _msg):
    	msg(_msg)
	{}

    virtual const char* what() const throw() {
        return (string("[Neural error] ") + msg).c_str();
    }
};

class UnimplementedException: public std::exception {
private:
    std::string msg;
public:
    UnimplementedException(const string& _msg):
    	msg(_msg)
	{}

    virtual const char* what() const throw() {
        return (string("[Unimplemented] ") + msg).c_str();
    }
};

} // end of anonymous namespace
#endif /* UTILS_H_ */
