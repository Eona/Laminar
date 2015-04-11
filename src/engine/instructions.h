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

	operator string() const
	{
		return name;
	}

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

	virtual ~Instruction() {}

	OpCode code;
	vector<int> readAddrs;
	int writeAddr;

	virtual operator string() const
	{
		return "{" + string(code) + ": "
				+ container2str(readAddrs) + " -> "
				+ to_str(writeAddr)
				+ "}";
	}
};

ostream& operator<<(ostream& os, Instruction instr)
{
	os << string(instr);
	return os;
}

#endif /* INSTRUCTIONS_H_ */
