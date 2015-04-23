/*
 * Eona Studio (c) 2015
 */

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "network.h"

/**************************************
******* Cross-validation and testing *********
**************************************/

template<typename FloatT = float>
class EvaluatorBase
{
public:
	EvaluatorBase(Network::Ptr net) :
		net(net)
	{}

	virtual ~EvaluatorBase() {}

	virtual FloatT loss_value() = 0;

	TYPEDEF_PTR(EvaluatorBase);

protected:
	Network::Ptr net;
};

/**
 * Evaluate network performance for validation and testing
 */
template<typename EngineT, typename FloatT = float>
class Evaluator : public EvaluatorBase<FloatT>
{
LMN_STATIC_ASSERT_IS_BASE(EngineBase, EngineT, "Evaluator template arg");

public:
	Evaluator(Network::Ptr net) :
		EvaluatorBase<FloatT>(net),
		engine(net->get_engine<EngineT>())
	{}

	virtual ~Evaluator() {}

	virtual FloatT loss_value()
	{
//		return engine->element_at(*net->get_total_loss());
		return 0;
	}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(Evaluator)

protected:
	std::shared_ptr<EngineT> engine;
};



#endif /* EVALUATOR_H_ */
