/*
 * Eona Studio (c) 2015
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "utils.h"
#include "layer.h"

class Connection
{
public:
	Connection(Layer& _inLayer, Layer& _outLayer):
		inLayer(_inLayer), outLayer(_outLayer)
    {
		// TODO random number
		param = 3.73;
    }

	void forward()
	{

	}

	void backward()
	{

	}

private:
	float param;
	Layer& inLayer;
	Layer& outLayer;
};

#endif /* CONNECTION_H_ */
