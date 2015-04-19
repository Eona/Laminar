/*
 * Eona Studio (c) 2015
 */


#ifndef TEST_H_
#define TEST_H_

#include <gtest/gtest.h>
#include "../global_utils.h"
#include "../rand_utils.h"
#include "../timer.h"
#include "../connection.h"
#include "../full_connection.h"
#include "../gated_connection.h"
#include "../activation_layer.h"
#include "../loss_layer.h"
#include "../parameter.h"
#include "../lstm.h"
#include "../network.h"
#include "../gradient_check.h"

#include "../engine/tensor.h"
#include "../engine/tensor_ops.h"

#include "../backend/dummy/dummy_dataman.h"
#include "../backend/dummy/dummy_engine.h"

#include "../backend/vector/vector_dataman.h"
#include "../backend/vector/vector_engine.h"

using namespace std;

#define conn_full Connection::make<FullConnection>
#define conn_const Connection::make<ConstantConnection>
#define conn_gated Connection::make<GatedConnection>

#define EXPECTED_LAMINAR_FAILURE(stmt) \
	{ \
		bool is_expected_thrown = false; \
		try { stmt } \
		catch (LaminarException& e) { \
			cout << "Expected failure: " << e.what() << endl; \
			is_expected_thrown = true; \
		} \
		if (!is_expected_thrown) \
			throw std::logic_error( \
				"LaminarException should have been thrown, but no throwing detected."); \
	}

static constexpr const int DUMMY_DIM = 666;

#endif /* TEST_H_ */
