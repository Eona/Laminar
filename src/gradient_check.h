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

	net.reset();
	for (int i = 0; i < timeLength; ++i)
		net.forward_prop();
	for (int i = 0; i < timeLength; ++i)
		net.backward_prop();

	vector<float> analyticGrads;
	for (ParamContainerPtr param : net.paramContainers)
	{
		auto grads = param->paramGradients;
		analyticGrads.insert(analyticGrads.end(), grads.begin(), grads.end());
	}

	/****** perturb parameters matrices stored in connections ******/
	int agpt = 0; // point to analyticGrads
	for (ParamContainerPtr param : net.paramContainers)
	{
		for (int p = 0; p < param->size(); ++p)
		{
			net.reset(); // refresh network
			param->gradient_check_perturb(p, -perturb);
			for (int i = 0; i < timeLength; ++i)
				net.forward_prop();
			float lossMinus = net.lossLayer->totalLoss;
			param->gradient_check_restore();

			net.reset(); // refresh network
			param->gradient_check_perturb(p, +perturb);
			for (int i = 0; i < timeLength; ++i)
				net.forward_prop();
			float lossPlus = net.lossLayer->totalLoss;
			param->gradient_check_restore();

			float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

			assert_float_percent_eq(analyticGrads[agpt++], numericGrad, percentTol,
					"param analytic != numeric", "param gradcheck pass");
		}
	}

	assert(agpt == analyticGrads.size(), "analyticGrads not fully traversed");

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

#endif /* GRADIENT_CHECK_H_ */
