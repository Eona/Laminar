/*
 * Eona Studio (c) 2015
 */

#ifndef GRADIENT_CHECK_H_
#define GRADIENT_CHECK_H_

#include "network.h"
#include "laminar_utils.h"

/**
 * Main gradient checker function
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
template<typename EngineT, typename DataManagerT>
inline void gradient_check(Network& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	auto engine = net.get_engine<EngineT>();
	auto dataman = net.get_data_manager<DataManagerT>();

	int historyLength = 1;
//	 Save the full gradient history for debugging ONLY
	try {
		RecurrentNetwork& net_ = dynamic_cast<RecurrentNetwork&>(net);
		net_.init_max_temporal_skip(Layer::UNLIMITED_TEMPORAL_SKIP);
		historyLength = net_.history_length();
	}
	catch (std::bad_cast& err) { }

	net.execute("initialize");
	net.execute("forward");
	net.execute("backward");

	// helper
	auto reset_net = [&]()
	{
		net.execute("zero_clear");
		dataman->start_new_epoch();
		// forward loads input, backward loads target,
		// but backward isn't called here, so we manually load_target
		net.execute("load_target");
	};

	vector<Tensor::Ptr> analyticGrads;
	for (ParamContainer::Ptr container : net.paramContainers)
	{
		for (int pidx = 0; pidx < container->size(); ++pidx)
			analyticGrads.push_back(
					Tensor::make(*container->param_gradient_ptr(pidx)));
	}
	engine->flush_execute();
//	for (auto& tensor : analyticGrads)
//		DEBUG_MSG("gradient check: " << engine->read_memory(tensor));

	/****** perturb parameters matrices stored in connections ******/
	int agpt = 0; // point to analyticGrads
	for (ParamContainer::Ptr param : net.paramContainers)
	{
		for (int p = 0; p < param->size(); ++p)
		{
			reset_net();
			param->gradient_check_perturb(p, {}, -perturb);
			engine->flush_execute();

			net.execute("forward");

			float lossMinus = *engine->read_memory(net.lossLayer->total_loss());

			param->gradient_check_restore();
			reset_net();

			param->gradient_check_perturb(p, {}, +perturb);
			engine->flush_execute();

			net.execute("forward");

			float lossPlus = *engine->read_memory(net.lossLayer->total_loss());

			param->gradient_check_restore();
			engine->flush_execute();

			float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

			float analyticGrad = *engine->read_memory(analyticGrads[agpt++]);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"param analytic != numeric", "param gradcheck pass");
		}
	}

	/****** perturb the input if GradientCheckable ******/
	reset_net();

	// If the given data manager implmenets GradientCheckable<FloatT>
	if (is_gradient_checkable<DataManagerT>::value)
	{
		net.execute("forward");
		net.execute("backward");

		analyticGrads.clear();
		for (int t = 0; t < historyLength; ++t)
			analyticGrads.push_back(
					Tensor::make(net.layers[0]->in_gradient(t)));

		engine->flush_execute();

		// perturb each input in sequence
		for (int inp = 0; inp < historyLength; ++inp)
		{
			reset_net();
			dataman->gradient_check_perturb(inp, {}, -perturb);

			net.execute("forward");

			float lossMinus = *engine->read_memory(net.lossLayer->total_loss());

			dataman->gradient_check_restore();
			reset_net();

			dataman->gradient_check_perturb(inp, {}, +perturb);

			net.execute("forward");

			float lossPlus = *engine->read_memory(net.lossLayer->total_loss());

			dataman->gradient_check_restore();

			float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

			float analyticGrad = *engine->read_memory(analyticGrads[inp]);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"input analytic != numeric", "input gradcheck pass");

		}
		reset_net();
	} // endif is_gradient_checkable<DataManager>
}

#endif /* GRADIENT_CHECK_H_ */
