/*
 * Eona Studio (c) 2015
 */


#ifndef INSTRUCTIONS_H_
#define INSTRUCTIONS_H_

#include "../global_utils.h"

struct Opcode
{
	Opcode(string _name) :
		name(_name)
	{ }

	Opcode(const char *_name) :
		name(_name)
	{ }

	operator string() const
	{
		return name;
	}

	bool operator==(const string str) const
	{
		return name == str;
	}

	bool operator!=(const string str) const
	{
		return !operator==(str);
	}

	string name;
};

/**
 * Hashing
 */
namespace std {
  template<>
  struct hash<Opcode>
  {
    size_t operator()(const Opcode& opcode) const
    {
    	// WARNING don't forget double ()!
    	return std::hash<std::string>()(opcode.name);
    }
  };
}

// Forward declare for OpContextBase
template<typename ...ContextArgT>
struct OpContext;

/**
 * Operation context
 */
struct OpContextBase
{
	virtual ~OpContextBase() {}
	typedef shared_ptr<OpContextBase> Ptr;

	/**
	 * Down cast to a specific context
	 */
	template<typename ...ContextArgT>
	static shared_ptr<OpContext<ContextArgT...>> cast(Ptr contextBase)
	{
		return std::dynamic_pointer_cast<OpContext<ContextArgT...>>(contextBase);
	}
};

template<typename ...ContextArgT>
struct OpContext : OpContextBase
{
	OpContext(ContextArgT ... args) :
		contextArgPack(std::make_tuple(args...))
	{ }

	std::tuple<ContextArgT...>& get_context_arg_pack()
	{
		return this->contextArgPack;
	}

	template<typename ...ArgT>
	static OpContextBase::Ptr make(ArgT&& ... args)
	{
		return static_cast<OpContextBase::Ptr>(
				std::make_shared<OpContext<ContextArgT...>>(
						std::forward<ArgT>(args) ...));
	}

private:
	std::tuple<ContextArgT...> contextArgPack;
};


struct Instruction
{
	Instruction(Opcode _code, vector<int>& _readAddrs, int _writeAddr) :
		opcode(_code), readAddrs(_readAddrs), writeAddr(_writeAddr)
	{ }

	Instruction(Opcode _code, const std::initializer_list<int>& _readAddrs, int _writeAddr) :
		opcode(_code), readAddrs(_readAddrs), writeAddr(_writeAddr)
	{ }

	virtual ~Instruction() {}

	Opcode opcode;
	vector<int> readAddrs;
	int writeAddr;
	OpContextBase::Ptr context;

	virtual operator string() const
	{
		return "{" + string(opcode) + ": "
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
