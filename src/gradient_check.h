/*
 * Eona Studio (c) 2015
 */

#ifndef GRADIENT_CHECK_H_
#define GRADIENT_CHECK_H_

#include "network.h"

/**
 * Check if a given class is a subclass of GradientCheckable<FloatT> for any FloatT
 */
GEN_IS_DERIVED_TEMPLATE_TRAIT(is_gradient_checkable, GradientCheckable);

/**
 * Main gradient checker function
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
template<typename EngineT, typename DataManagerT, typename FloatT = float>
inline void gradient_check(Network& net,
		FloatT perturb = 1e-2f, FloatT percentTol = 1.0f)
{
	LAMINAR_STATIC_ASSERT((std::is_base_of<ElementInspectionBase, EngineT>::value),
		"Engine must implement ElementInspection<> interface to work with gradient_check");

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

	// helper
	// loss "tensor" is always a 1x1 Scalor
	auto read_loss = [&]() -> FloatT
	{
		return engine->element_at(
			engine->read_memory(net.lossLayer->total_loss()), {0, 0});
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
	// step through every parameter container
	for (ParamContainer::Ptr param : net.paramContainers)
	{
		// step through every parameter in the container
		for (int p = 0; p < param->size(); ++p)
		{
			// step through every element entry in the parameter tensor
			Dimension totalParamDim = param->param_dim(p);
			DimIndexEnumerator idxEnumer(totalParamDim);

			while (idxEnumer.has_next())
			{
				DimIndex perturbDimIdx = idxEnumer.next();

				reset_net();
				param->gradient_check_perturb(p, perturbDimIdx, -perturb);
				engine->flush_execute();

				net.execute("forward");

				FloatT lossMinus = read_loss();

				param->gradient_check_restore();
				reset_net();

				param->gradient_check_perturb(p, perturbDimIdx, +perturb);
				engine->flush_execute();

				net.execute("forward");

				FloatT lossPlus = read_loss();

				param->gradient_check_restore();
				engine->flush_execute();

				FloatT numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

				FloatT analyticGrad = engine->element_at(
						engine->read_memory(analyticGrads[agpt++]), perturbDimIdx);

				assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
						"param analytic != numeric", "param gradcheck pass");
			}
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

		// perturb each input tensor in sequence
		for (int inp = 0; inp < historyLength; ++inp)
		{
			// step through every element entry in the input tensor
			Dimension totalInputDim = dataman->input_dim();
			DimIndexEnumerator idxEnumer(totalInputDim);
			while (idxEnumer.has_next())
			{
				DimIndex perturbDimIdx = idxEnumer.next();

				reset_net();
				dataman->gradient_check_perturb(inp, perturbDimIdx, -perturb);

				net.execute("forward");

				FloatT lossMinus = read_loss();

				dataman->gradient_check_restore();
				reset_net();

				dataman->gradient_check_perturb(inp, perturbDimIdx, +perturb);

				net.execute("forward");

				FloatT lossPlus = read_loss();

				dataman->gradient_check_restore();

				FloatT numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

				FloatT analyticGrad = engine->element_at(
						engine->read_memory(analyticGrads[inp]), perturbDimIdx);

				assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
						"input analytic != numeric", "input gradcheck pass");
			}
		}
		reset_net();
	} // endif is_gradient_checkable<DataManager>
}

#endif /* GRADIENT_CHECK_H_ */
