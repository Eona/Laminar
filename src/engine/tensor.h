/*
 * Eona Studio (c) 2015
 */


#ifndef TENSOR_H_
#define TENSOR_H_

#include "../global_utils.h"
#include "engine.h"

struct TensorBase
{
	TensorBase(EngineBase::Ptr _engine) :
		engine(_engine)
	{
		this->addr = engine->alloc();
		engine->upload(Instruction("create", {}, addr));
	}

	virtual ~TensorBase()
	{
		engine->upload(Instruction("destroy", {}, addr));
		this->addr = -1;
	}

	EngineBase::Ptr engine;
	// memory address in the engine, if negative -> destroyed
	int addr;
};

struct Tensor : public TensorBase
{
	Tensor(EngineBase::Ptr _engine) :
		TensorBase(_engine)
	{ }

	virtual ~Tensor() {}

};

struct Scalor : public TensorBase
{
	Scalor(EngineBase::Ptr _engine) :
		TensorBase(_engine)
	{ }

	virtual ~Scalor() {}
};

/**
 * Only Tensor + Tensor or Scalor + Scalor
 */
template<typename TensorT>
typename std::enable_if<std::is_base_of<TensorBase, TensorT>::value, Tensor>::type
operator+(const TensorT& t1, const TensorT& t2)
{
	TensorT ans(t1.engine);
	t1.engine->upload(Instruction("add", {t1.addr, t2.addr}, ans.addr));
	return ans;
}

/**
 * Only Tensor - Tensor or Scalor - Scalor
 */
template<typename TensorT>
typename std::enable_if<std::is_base_of<TensorBase, TensorT>::value, Tensor>::type
operator-(const TensorT& t1, const TensorT& t2)
{
	TensorT ans(t1.engine);
	t1.engine->upload(Instruction("subtract", {t1.addr, t2.addr}, ans.addr));
	return ans;
}


#endif /* TENSOR_H_ */
