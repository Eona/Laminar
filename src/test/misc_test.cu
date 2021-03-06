/*
 * Eona Studio (c) 2015
 *
 * Miscellaneous tests
 */

#include "test_utils.h"

TEST(Composite, NetworkMismatch)
{
	ForwardNetwork net;

	auto lstm = Composite<RecurrentNetwork>
			::create<LstmComposite>(Layer::make<ConstantLayer>());

	LMN_EXPECTED_FAILURE(
		net.add_composite(lstm);
	)
}
