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
	net.execute("load_input"); net.execute("load_target");
	net.execute("forward");
	net.execute("backward");

	// helper
	auto reset_net = [&]()
	{
		net.execute("zero_clear");
		dataman->start_new_epoch();
		net.execute("load_input"); net.execute("load_target");
	};

	// helper
	// loss "tensor" is always a 1x1 Scalor
	auto read_loss = [&]() -> FloatT
	{
		return engine->scalor_at(net.lossLayer->total_loss());
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

	/**************************************
	******* Perturb every element in the parameter tensors *********
	**************************************/
	// total number of elements in all parameter tensors combined
	int totalParamElementCount = 0;

	int agpt = 0; // point to analyticGrads
	// step through every parameter container
	for (ParamContainer::Ptr param : net.paramContainers)
	{
		// step through every parameter in the container
		for (int p = 0; p < param->size(); ++p)
		{
			// step through every element entry in the parameter tensor
			Dimension paramDim = param->param_dim(p);
			DimIndexEnumerator idxEnumer(paramDim);
			totalParamElementCount +=
					std::accumulate(paramDim.begin(), paramDim.end(), 1, std::multiplies<int>());

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

				FloatT analyticGrad = engine->tensor_at(analyticGrads[agpt], perturbDimIdx);

				assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
						"param analytic != numeric", "param gradcheck pass");
			}
			++ agpt; // go to the next tensor stored in analyticGrads
		}
	}
	cout << "Total parameter element count: " << totalParamElementCount << endl;

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

		// Cast gauranteed to succeed because we've checked is_gradient_checkable<>
		// in the enclosing if() block.
		auto datamanGradCheck =
				std::dynamic_pointer_cast<GradientCheckable<FloatT> >(dataman);

		LMN_ASSERT_NULLPTR(datamanGradCheck,
				LaminarException("DataManager type doesn't implement GradientCheckable<FloatT>"));

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
				datamanGradCheck->gradient_check_perturb(inp, perturbDimIdx, -perturb);

				// reload input data after perturbation
				dataman->start_new_epoch(); net.execute("load_input"); net.execute("load_target");
				net.execute("forward");

				FloatT lossMinus = read_loss();

				datamanGradCheck->gradient_check_restore();
				reset_net();

				datamanGradCheck->gradient_check_perturb(inp, perturbDimIdx, +perturb);

				// reload input data after perturbation
				dataman->start_new_epoch(); net.execute("load_input"); net.execute("load_target");
				net.execute("forward");

				FloatT lossPlus = read_loss();

				datamanGradCheck->gradient_check_restore();

				FloatT numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

				FloatT analyticGrad = engine->tensor_at(analyticGrads[inp], perturbDimIdx);

				assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
						"input analytic != numeric", "input gradcheck pass");
			}
		}
		reset_net();
	} // endif is_gradient_checkable<DataManager>
}

#endif /* GRADIENT_CHECK_H_ */
