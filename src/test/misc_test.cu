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

	try {
		net.add_composite(lstm);
	}
	catch (LaminarException& e)
	{
		cout << "Expected failure: " << e.what() << endl;
	}
}
