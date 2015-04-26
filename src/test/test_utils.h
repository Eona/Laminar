/*
 * Eona Studio (c) 2015
 */


#ifndef TEST_H_
#define TEST_H_

#include <gtest/gtest.h>
#include "../connection.h"
#include "../full_connection.h"
#include "../gated_connection.h"
#include "../activation_layer.h"
#include "../bias_layer.h"
#include "../loss_layer.h"
#include "../parameter.h"
#include "../network.h"
#include "../rnn.h"
#include "../lstm.h"
#include "../gradient_check.h"

#include "../engine/tensor.h"
#include "../engine/tensor_ops.h"

#include "../backend/dummy/dummy_dataman.h"
#include "../backend/dummy/dummy_engine.h"

#include "../backend/vecmat/vecmat_engine.h"
#include "../backend/vecmat/vecmat_rand_dataman.h"
#include "../backend/vecmat/vecmat_func_dataman.h"
#include "rand_dataman.h"

using namespace std;

#define conn_full Connection::make<FullConnection>
#define conn_const Connection::make<ConstantConnection>
#define conn_gated Connection::make<GatedConnection>

#define LMN_EXPECTED_FAILURE(stmt) \
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

static constexpr const int DUMMY_DIM = 1;

#endif /* TEST_H_ */
