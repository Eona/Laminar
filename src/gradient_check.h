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
inline void gradient_check(ForwardNetwork& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	/****** perturb parameters matrices stored in connections ******/
	for (ConnectionPtr conn : net.connections)
	{
		net.reset(); // refresh network

		auto linearConn = cast_connection<LinearConnection>(conn);
		auto constConn = cast_connection<ConstantConnection>(conn);
		if (linearConn)
		{
			net.forward_prop();
			net.backward_prop();
			float analyticGrad = linearConn->gradient;
			float oldParam = linearConn->param; // for restoration

			// perturb the parameter and run again
			net.reset();
			linearConn->param = oldParam - perturb;
			net.forward_prop();
			float lossMinus = net.lossLayer->totalLoss;

			net.reset();
			linearConn->param = oldParam + perturb;
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
	float oldInput = net.input; // for restoration

	net.reset();
	net.forward_prop();
	net.backward_prop();
	float analyticGrad = net.layers[0]->inGradient[0];

	net.set_input(oldInput - perturb);
	net.reset();
	net.forward_prop();
	float lossMinus = net.lossLayer->totalLoss;

	net.set_input(oldInput + perturb);
	net.reset();
	net.forward_prop();
	float lossPlus = net.lossLayer->totalLoss;

	float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

	assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
			"input analytic != numeric", "input gradcheck pass");
}


/**
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
inline void gradient_check(RecurrentNetwork& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	/****** perturb parameters matrices stored in connections ******/
	for (ConnectionPtr conn : net.connections)
	{
		net.reset(); // refresh network

		auto linearConn = cast_connection<LinearConnection>(conn);
		auto constConn = cast_connection<ConstantConnection>(conn);
		if (linearConn)
		{
			net.forward_prop();
			net.backward_prop();
			float analyticGrad = linearConn->gradient;
			float oldParam = linearConn->param; // for restoration

			// perturb the parameter and run again
			net.reset();
			linearConn->param = oldParam - perturb;
			net.forward_prop();
			float lossMinus = net.lossLayer->totalLoss;

			net.reset();
			linearConn->param = oldParam + perturb;
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
/*	vector<float> oldInput = net.input; // for restoration

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
			"input analytic != numeric", "input gradcheck pass");*/
}

#endif /* GRADIENT_CHECK_H_ */
