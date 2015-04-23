/*
 * Eona Studio (c) 2015
 */


#ifndef INSTRUCTIONS_H_
#define INSTRUCTIONS_H_

#include "../utils/global_utils.h"
#include "../utils/laminar_utils.h"
#include "../utils/debug_utils.h"

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

	TYPEDEF_PTR(OpContextBase);

	/**
	 * Down cast to a specific context
	 */
	template<typename ...ContextArgT>
	static std::shared_ptr<OpContext<ContextArgT...>> cast(Ptr contextBase)
	{
		return std::dynamic_pointer_cast<OpContext<ContextArgT...>>(contextBase);
	}

	template<typename ...ContextArgT>
	static OpContextBase::Ptr make(ContextArgT&& ... args)
	{
		return std::static_pointer_cast<OpContextBase>(
				std::make_shared<OpContext<ContextArgT...>>(
						std::forward<ContextArgT>(args) ...));
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
		return std::static_pointer_cast<OpContextBase>(
				std::make_shared<OpContext<ContextArgT...>>(
						std::forward<ArgT>(args) ...));
	}

private:
	std::tuple<ContextArgT...> contextArgPack;
};


struct Instruction
{
	Instruction(Opcode code_, vector<int> readAddrs_, int writeAddr_,
			OpContextBase::Ptr context_ = nullptr) :
		opcode(code_), readAddrs(readAddrs_), writeAddr(writeAddr_),
		context(context_)
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
				+ (bool(context) ? " context" : "")
				+ "}";
	}
};

std::ostream& operator<<(std::ostream& os, Instruction instr)
{
	os << string(instr);
	return os;
}

typedef std::function<void()> Executable;

/**
 * Contains a sequence of Instructions and possibly the compiled 'executable'
 * Executables are stored as void() lambdas
 */
struct Routine
{
	Routine(vector<Instruction> instructions_ = {},
			vector<Executable> executables_ = {}):
		instructions(instructions_), executables(executables_)
	{ }

	vector<Instruction> instructions;
	vector<Executable> executables;

	inline void operator()() const
	{
		this->execute();
	}

	void execute() const
	{
		LMN_ASSERT_THROW(!this->instructions.empty(),
			EngineException("Routine contains no instructions, cannot be executed."));

		LMN_ASSERT_THROW(this->is_compiled(),
			EngineException("Routine has not been compiled yet."));

		for (Executable exe : executables)
			exe();
	}

	bool is_compiled() const
	{
		return !executables.empty();
	}

	/************************************/
	TYPEDEF_PTR(Routine);

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(Routine)
};

#endif /* INSTRUCTIONS_H_ */
