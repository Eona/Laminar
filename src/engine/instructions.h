/*
 * Eona Studio (c) 2015
 */


#ifndef INSTRUCTIONS_H_
#define INSTRUCTIONS_H_

#include "../global_utils.h"

struct OpCode
{
	OpCode(string _name) :
		name(_name)
	{ }

	OpCode(const char *_name) :
		name(_name)
	{ }

	string name;
};

struct Instruction
{
	Instruction(OpCode _code, vector<int>& _readAddrs, int _writeAddr) :
		code(_code), readAddrs(_readAddrs), writeAddr(_writeAddr)
	{ }

	Instruction(OpCode _code, const std::initializer_list<int>& _readAddrs, int _writeAddr) :
		code(_code), readAddrs(_readAddrs), writeAddr(_writeAddr)
	{ }

	OpCode code;
	vector<int> readAddrs;
	int writeAddr;
};


#endif /* INSTRUCTIONS_H_ */
