/*
 * Eona Studio (c) 2015
 */

#ifndef GRADIENT_CHECK_H_
#define GRADIENT_CHECK_H_

#include "network.h"

/**
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
template<typename NetworkT>
inline void gradient_check(NetworkT& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	// if feedforward net, the length is simply 1
	using InputType = typename NetworkT::InputType;
	bool isForward = !is_vector<InputType>::value;
//	int timeLength = isForward ? 1 : net.input.size();
	int timeLength = 4;

	/****** perturb parameters matrices stored in connections ******/
	for (ConnectionPtr conn : net.connections)
	{
		net.reset(); // refresh network

		auto linearConn = cast_connection<LinearConnection>(conn);
		auto constConn = cast_connection<ConstantConnection>(conn);
		if (linearConn)
		{
			for (int i = 0; i < timeLength; ++i)
				net.forward_prop();
			for (int i = 0; i < timeLength; ++i)
				net.backward_prop();
			float analyticGrad = linearConn->gradient;
			float oldParam = linearConn->param; // for restoration

			// perturb the parameter and run again
			net.reset();
			linearConn->param = oldParam - perturb;
			for (int i = 0; i < timeLength; ++i)
				net.forward_prop();
			float lossMinus = net.lossLayer->totalLoss;

			net.reset();
			linearConn->param = oldParam + perturb;
			for (int i = 0; i < timeLength; ++i)
				net.forward_prop();
			float lossPlus = net.lossLayer->totalLoss;

			float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"param analytic != numeric", "param gradcheck pass");

			linearConn->param = oldParam;
		}
		else if (constConn) { }
	}

	/****** perturb the input ******/
	InputType oldInput = net.input; // for restoration

	net.reset();
	for (int i = 0; i < timeLength; ++i)
		net.forward_prop();
	for (int i = 0; i < timeLength; ++i)
		net.backward_prop();

	vector<float> analyticGrad = net.layers[0]->inGradient;
	vector<float> numericGrad(timeLength);

/*	net.set_input(oldInput - perturb);
	net.reset();
	net.forward_prop();
	float lossMinus = net.lossLayer->totalLoss;

	net.set_input(oldInput + perturb);
	net.reset();
	net.forward_prop();
	float lossPlus = net.lossLayer->totalLoss;*/

//	float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);


	for (int i = 0; i < timeLength; ++i)
		assert_float_percent_eq(analyticGrad[i], numericGrad[i], percentTol,
				"input analytic != numeric", "input gradcheck pass");
}

/*
*
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients

inline void gradient_check(RecurrentNetwork& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	***** perturb parameters matrices stored in connections *****
	for (ConnectionPtr conn : net.connections)
	{
		net.reset(); // refresh network

		auto linearConn = cast_connection<LinearConnection>(conn);
		auto constConn = cast_connection<ConstantConnection>(conn);
		if (linearConn)
		{
			net.forward_prop();
			net.forward_prop();
			net.forward_prop();
			net.backward_prop();
			net.backward_prop();
			net.backward_prop();
//			DEBUG_MSG("orig net", net);
			float analyticGrad = linearConn->gradient;
			float oldParam = linearConn->param; // for restoration

			// perturb the parameter and run again
			net.reset();
			linearConn->param = oldParam - perturb;

			net.forward_prop();
			net.forward_prop();
			net.forward_prop();
//			DEBUG_MSG("forward net", net);
			float lossMinus = net.lossLayer->totalLoss;
//			DEBUG_MSG("lossMinus", lossMinus);

			net.reset();
			linearConn->param = oldParam + perturb;
			net.forward_prop();
			net.forward_prop();
			net.forward_prop();
//			DEBUG_MSG("forward net", net);
			float lossPlus = net.lossLayer->totalLoss;
//			DEBUG_MSG("lossPlus", lossPlus);

			float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"param analytic != numeric", "param gradcheck pass");

			linearConn->param = oldParam;
		}
		else if (constConn) { }
	}

	***** perturb the input *****
	vector<float> oldInput = net.input; // for restoration

	net.reset();
	net.forward_prop();
	net.backward_prop();
	float analyticGrad = net.layers[0]->inGradient[0];

//	net.set_input(oldInput - perturb);
	net.reset();
	net.forward_prop();
	float lossMinus = net.lossLayer->totalLoss;

//	net.set_input(oldInput + perturb);
	net.reset();
	net.forward_prop();
	float lossPlus = net.lossLayer->totalLoss;

	float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

	assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
			"input analytic != numeric", "input gradcheck pass");
}*/

#endif /* GRADIENT_CHECK_H_ */
