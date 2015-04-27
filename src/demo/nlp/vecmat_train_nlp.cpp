/*
 * Eona Studio (c) 2015
 */

#include "../../full_connection.h"
#include "../../loss_layer.h"
#include "../../activation_layer.h"
#include "../../bias_layer.h"
#include "../../parameter.h"
#include "../../network.h"
#include "../../learning_session.h"
#include "../../utils/rand_utils.h"

#include "../../backend/vecmat/vecmat_engine.h"
#include "corpus_loader.h"

int main(int argc, char **argv)
{
	CorpusLoader corpus("../data/corpus/shakespeare_midsummer.dat");
	int s = corpus.size();
	DEBUG_MSG(s);

	s -= s % 100;
	for (int b = 0; b < s / 1000 - 1; ++b)
		corpus.load(1000);


	DEBUG_MSG(CorpusLoader::convert(corpus.load(1000)));
}


