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
inline void gradient_check(Network& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	int timeLength = net.input.size();
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
	vector<float> oldInput = net.input; // for restoration

	net.reset();
	for (int i = 0; i < timeLength; ++i)
		net.forward_prop();
	for (int i = 0; i < timeLength; ++i)
		net.backward_prop();

	vector<float> analyticGrad = net.layers[0]->inGradient;
	vector<float> numericGrad(net.input.size());

	// perturb each input in sequence
	for (int inp = 0; inp < timeLength; ++inp)
	{
		float restoreInputVal = oldInput[inp];

		oldInput[inp] = restoreInputVal - perturb;
		net.set_input(oldInput);
		net.reset();
		for (int i = 0; i < timeLength; ++i)
			net.forward_prop();
		float lossMinus = net.lossLayer->totalLoss;

		oldInput[inp] = restoreInputVal + perturb;
		net.set_input(oldInput);
		net.reset();
		for (int i = 0; i < timeLength; ++i)
			net.forward_prop();
		float lossPlus = net.lossLayer->totalLoss;

		float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

		assert_float_percent_eq(analyticGrad[inp], numericGrad, percentTol,
				"input analytic != numeric", "input gradcheck pass");

		oldInput[inp] = restoreInputVal; // restore
	}
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
