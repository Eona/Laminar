/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECTOR_VECTOR_ENGINE_H_
#define BACKEND_VECTOR_VECTOR_ENGINE_H_



/*
class DummyEngine : public Engine<float>
{
public:
	DummyEngine() :
		Engine<float>()
	{
		namespace Impl = lmn::DummyImpl;
		const int T = Impl::TENSOR;
		const int S = Impl::SCALOR;

		register_create_op(Impl::create);

		register_normal_op("t+t", Impl::add<T>);
		register_normal_op("s+s", Impl::add<S>);
		register_normal_op("t-t", Impl::sub<T>);
		register_normal_op("s-s", Impl::sub<S>);
		register_normal_op("-t", Impl::negate<T>);
		register_normal_op("-s", Impl::negate<S>);
		register_normal_op("t*t", Impl::mult<T, T>);
		register_normal_op("t*s", Impl::mult<T, S>);
		register_normal_op("s*t", Impl::mult<S, T>);
		register_normal_op("s*s", Impl::mult<S, S>);
		register_normal_op("t=t", Impl::assign<T>);
		register_normal_op("s=s", Impl::assign<S>);

		register_normal_op("sin", Impl::sin);
		register_normal_op("cos", Impl::cos);
		register_normal_op("tanh", Impl::tanh);
		register_normal_op("tanh_gradient", Impl::tanh_gradient);
		register_normal_op("sigmoid", Impl::sigmoid);
		register_normal_op("sigmoid_gradient", Impl::sigmoid_gradient);
		register_normal_op("transpose", Impl::transpose);
		register_normal_op("element_mult", Impl::element_mult);
		register_normal_op("square_loss", Impl::square_loss);

		register_normal_op("destroy", Impl::destroy);
		register_normal_op("clear", Impl::clear);

		register_normal_op("fill_rand", Impl::fill_rand);
		register_normal_op("fill_rand_prehistory", Impl::fill_rand_prehistory);
		register_context_op<DimIndex, float>("perturb", Impl::perturb);
		register_context_op<DimIndex, float>("set_value", Impl::set_value);
		register_context_op<float>("scale", Impl::scale);

		********** DEBUG ONLY **********
		register_context_op<string, float, std::pair<char, int>>("debug_context_tmp", Impl::debug_context_tmp);
	}
};*/


#endif /* BACKEND_VECTOR_VECTOR_ENGINE_H_ */
