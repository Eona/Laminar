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
#include "../transfer_layer.h"
#include "../loss_layer.h"
#include "../parameter.h"
#include "../lstm.h"
#include "../network.h"
#include "../gradient_check.h"

using namespace std;

#define conn_full Connection::make<FullConnection>
#define conn_const Connection::make<ConstantConnection>
#define conn_gated Connection::make<GatedConnection>

#endif /* TEST_H_ */
