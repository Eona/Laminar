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
#include <sstream>
#include <algorithm>
#include <functional>
#include <climits>
#include <cstdarg>
#include <utility>
#include <type_traits>
#include <tuple>

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
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
std::ostream& operator<<(std::ostream& oss, const vector<T>& vec)
{
	return oss << container2str(vec);
}
template<typename T>
std::ostream& operator<<(std::ostream& oss, vector<T>&& vec)
{
	return oss << vec;
}

template<typename T>
inline string to_str(T val)
{
	std::ostringstream oss;
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
inline void vec_for_each(vector<T>& vec, Func f)
{
	std::for_each(vec.begin(), vec.end(), f);
}

template<typename T>
inline vector<T> vec_augment(const vector<T>& vec, const T& extra)
{
	vector<T> ans = vec;
	ans.push_back(extra);
	return ans;
}

template <typename KeyT, typename ValueT>
bool key_exists(std::unordered_map<KeyT, ValueT>& map, KeyT& key)
{
	return map.find(key) != map.end();
}

// enclose x as "x" in macro expansion
#define STRINGFY_(x) #x
#define STRINGFY(x) STRINGFY_(x)

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

/**
 * Template sorcery for unpacking tuples to be function args
 * deferred_func_call is an example of how to unpack
 * http://stackoverflow.com/questions/7858817/unpacking-a-tuple-to-call-a-matching-function-pointer
 */
template<int ...>
struct unpack_seq {};

template<int N, int ...S>
struct unpack_gens : unpack_gens<N-1, N-1, S...> {};

template<int ...S>
struct unpack_gens<0, S...>{ typedef unpack_seq<S...> type; };

// deferred_func_call<FunctionReturnType, ArgTypes...>
template <typename ReturnT, typename ...ArgT>
struct deferred_func_call
{
	typedef std::function<ReturnT(ArgT...)> FuncType;
	typedef std::tuple<ArgT...> ArgPack;

	deferred_func_call() {}

	deferred_func_call(FuncType func_, ArgPack args_) :
		func(func_), args(args_)
	{ }

	void set_func(FuncType func)
	{
		this->func = func;
	}

	void set_args(ArgPack args)
	{
		this->args = args;
	}

	ReturnT operator()()
	{
		if (!func)
			throw std::runtime_error("No function saved in deferred_func_call");
		return call_helper(typename unpack_gens<sizeof...(ArgT)>::type());
	}

private:
	FuncType func;
	ArgPack args;

	template<int ...UnpackS>
	ReturnT call_helper(unpack_seq<UnpackS...>)
	{
		return func(std::get<UnpackS>(args) ...);
	}
};
} // end of anonymous namespace
#endif /* UTILS_H_ */
