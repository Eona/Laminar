/*
 * Eona Studio (c) 2015
 */


#ifndef ENGINE_H_
#define ENGINE_H_

#include "instructions.h"

template<typename DataT>
class MemoryPool
{
LMN_STATIC_ASSERT(std::is_default_constructible<DataT>::value,
		"Data type in MemoryPool must be default constructible.");

public:
	MemoryPool()
	{ }

	virtual ~MemoryPool() {};

	typedef std::shared_ptr<DataT> DataPtr;

	int alloc()
	{
		this->memory.push_back(DataPtr(new DataT()));
		this->initialized.push_back(false);
		return size() - 1;
	}

	DataPtr operator[](int i)
	{
		return memory[i];
	}

	bool is_initialized(int i)
	{
		return this->initialized[i];
	}

	void set_initialized(int i, bool val = true)
	{
		this->initialized[i] = val;
	}

	int size()
	{
		return memory.size();
	}

	void reset()
	{
		this->memory.clear();
		this->initialized.clear();
	}

	template<typename T>
	friend std::ostream& operator<<(std::ostream& os, MemoryPool<T>& memoryPool);

private:
	// All the following should have the same size
	vector<DataPtr> memory;
	// test if things are default initialized.
	vector<bool> initialized;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, MemoryPool<T>& memoryPool)
{
	os << "MemoryPool" << memoryPool.memory;
	return os;
}

/*********** Expression tree ***********/
// TensorNode only contains an int address to the memory pool
// It's agnostic of Tensor
struct TensorNode
{
	// Use static ::make to construct
	TensorNode(int _addr) :
		addr(_addr)
	{}

	TensorNode(const TensorNode&) =delete;
	TensorNode& operator=(const TensorNode&) =delete;
	TensorNode(TensorNode&&) =delete;
	TensorNode& operator=(TensorNode&&) =delete;

	bool has_child() const
	{
		return !this->children.empty();
	}

	bool has_parent() const
	{
		return !this->parents.empty();
	}

	typedef std::shared_ptr<TensorNode> Ptr;

	static TensorNode::Ptr make(int addr)
	{
		return std::make_shared<TensorNode>(addr);
	}

	static TensorNode::Ptr create_alias(const TensorNode::Ptr other)
	{
		return std::make_shared<TensorNode>(other->addr);
	}

	explicit operator string() const
	{
		return string("Node{")
			+ to_str(addr) + ", "
			+ "child=" + node_vec_str(this->children) + ", "
			+ "parent=" + node_vec_str(this->parents)
			+ "}";
	}

	int addr; // real memory address in Engine
	vector<Ptr> children;
	vector<Ptr> parents;

private:
	string node_vec_str(const vector<Ptr>& vec) const
	{
		string s = "[";
		for (Ptr p : vec)
			s += to_str(p->addr) + ", ";
		return (s.size() > 1 ?
				s.substr(0, s.size() - 2) : s) + "]";
	}
};

std::ostream& operator<<(std::ostream& os, TensorNode& node)
{
	os << string(node);
	return os;
}


/**
 * Responsible for storing the meta-instructions and
 * optimize the instruction graph with compiler techniques.
 */
class EngineBase
{
public:
	EngineBase() :
		currentRoutine(Routine::make())
	{ }

	virtual ~EngineBase() {};

	void upload(Instruction instr)
	{
		currentRoutine->instructions.push_back(instr);
	}

	virtual int alloc() = 0;

	virtual void reset() = 0;

	/**
	 * Construct a DAG of data dependencies
	 */
	virtual void construct_graph()
	{
		// TODO only deal with current routine?
		for (Instruction& instr : currentRoutine->instructions)
		{
			vector<int>& reads = instr.readAddrs;
			int write = instr.writeAddr;
			if(starts_with(instr.opcode, "create"))
			{
				// The node is stored at the same int index as the memory pool
				this->createdNodes.push_back(TensorNode::make(write));
			}
			else
			{
				// If the write node is not freshly created (has children),
				// it's an assignment and we create a new node alias
				// with the same internal memory addr
				TensorNode::Ptr node = this->createdNodes[write];
				if (node->has_child())
					node = TensorNode::create_alias(node);
				for (int read : reads)
				{
					node->children.push_back(this->createdNodes[read]);
					this->createdNodes[read]->parents.push_back(node);
				}
			}
		}
	}

	virtual void print_graph()
	{
		for (auto node : this->createdNodes)
			cout << string(*node) << "\n";
	}

	virtual void eliminate_temporary()
	{
		auto& instructions = currentRoutine->instructions;
		auto i = instructions.begin() + 3; // at least from 3rd instr onwards
		do {
			auto instr = *i;
			if (instr.opcode == "destroy")
			{
				auto instr_1 = i[-1]; // *(i - 1)
				auto instr_2 = i[-2];
				// instr_2 { t+t: [2, 2] -> 4 }
				// instr_1 { copy: [4] -> 3 }
				// instr { destroy: [] -> 4 }
				// optimize and eliminate temporary '4'
				// instr_new { t+t: [2, 2] -> 3 }
				if ((instr_1.opcode == "t=t" || instr_1.opcode == "s=s")
					&& instr_1.readAddrs[0] == instr.writeAddr
					&& instr_2.writeAddr == instr_1.readAddrs[0])
				{
					// instr_3 might have { create: [] -> 4 }
					auto instr_3 = i[-3];
					if (starts_with(instr_3.opcode, "create_null")
						&& instr_3.writeAddr == instr.writeAddr)
					{
						// eliminate all 4 instructions instr_3 ... instr, inclusive
						i = instructions.erase(i - 3, i + 1);
					}
					else
						i = instructions.erase(i - 2, i + 1);
					// Add new combined instr
					instructions.insert(i,
						Instruction(instr_2.opcode, instr_2.readAddrs, instr_1.writeAddr));
				}
			}
		}
		while (++i < instructions.end());
	}

	virtual void print_routines()
	{
		for (int i = 0; i < routines.size(); ++i)
		{
			cout << "$------------ Routine " << i << " ------------$\n";
			for (auto& instr : routines[i]->instructions)
				cout << instr << "\n";
		}
		if (!currentRoutine->instructions.empty())
		{
			cout << "$------------ Routine (unflushed) ------------$\n";
			for (auto& instr : currentRoutine->instructions)
				cout << instr << "\n";
		}
		else
			cout << "$------------ Routine (unflushed) empty\n";
	}

	/**************************************
	******* Routine management *********
	**************************************/
	/**
	 * Flush the currentRoutine to vector of routines and start a new routine.
	 * Later instructions will be pushed to a new routine
	 */
	Routine::Ptr flush()
	{
		Routine::Ptr flushedRoutine = currentRoutine;
		routines.push_back(currentRoutine);
		currentRoutine = Routine::make();
		return flushedRoutine;
	}

	/**
	 * Compile the routine.
	 * Store the executable inside the Routine.
	 * If the routine is already compiled, clear the executables and recompile.
	 */
	virtual void compile(Routine::Ptr) = 0;

	/**
	 * Shortcut: flush, compile and execute
	 * @return flushed routine
	 */
	Routine::Ptr flush_execute()
	{
		auto routine = this->flush();
		this->compile(routine);
		routine->execute();
		return routine;
	}

	/************************************/
	typedef std::shared_ptr<EngineBase> Ptr;

	GEN_GENERIC_MAKEPTR_STATIC_MEMBER(EngineBase)

	GEN_DOWN_CAST_STATIC_MEMBER(EngineBase)

protected:
	Routine::Ptr currentRoutine;

	vector<Routine::Ptr> routines;

	/**
	 * Only contains TensorNodes created by "create" opcode.
	 */
	vector<TensorNode::Ptr> createdNodes;
};

TYPEDEF_PTR_EXTERNAL(EngineBase);


/*********** Forward decl ***********/
// actual impl will be defined in tensor.h to avoid mutual header inclusion
class TensorBase;
class Scalor;

template<typename DataT, typename FloatT = float>
class Engine : public EngineBase
{
public:
	Engine() :
		EngineBase()
	{ }

	virtual ~Engine() {};

	typedef std::shared_ptr<Engine<DataT>> Ptr;
	typedef std::shared_ptr<DataT> DataPtr;

	virtual int alloc()
	{
		return memoryPool.alloc();
	}

	virtual void reset()
	{
		this->memoryPool.reset();
	}

	/**************************************
	******* Retrieve actual data from MemoryPool *********
	**************************************/
	/**
	 * Return ref to actual data
	 * NOTE will be defined in tensor.h to avoid mutual header inclusion
	 * @param tensorPtr
	 * @return
	 */
	DataPtr read_memory(std::shared_ptr<TensorBase>);
	DataPtr read_memory(const TensorBase&);

	/**
	 * Get a float element entry from the actual Tensor data
	 */
	virtual FloatT tensor_data_at(DataPtr, DimIndex) = 0;
	/**
	 * Get a float from Scalor
	 */
	virtual FloatT scalor_data_at(DataPtr) = 0;

	/**
	 * @return a float element entry from tensor
	 */
	FloatT tensor_at(std::shared_ptr<TensorBase> tensor, DimIndex idx)
	{
		return tensor_data_at(read_memory(tensor), idx);
	}
	FloatT tensor_at(const TensorBase& tensor, DimIndex idx)
	{
		return tensor_data_at(read_memory(tensor), idx);
	}

	FloatT scalor_at(std::shared_ptr<Scalor> scalor)
	{
		return scalor_data_at(
			read_memory(std::static_pointer_cast<TensorBase>(scalor)));
	}
	FloatT scalor_at(const Scalor& scalor)
	{
		// ugly cast workaround for forward decl
		return scalor_data_at(read_memory((TensorBase&) scalor));
	}

	/**************************************
	******* Register "assembly" commands ******
	**************************************/
	// (readAddrs, writeAddr, is_initialized)
	typedef std::function<void(vector<DataPtr>, DataPtr, bool)> CommandFuncType;
	// specifically for Opcode create(writeAddr, dim)
	typedef std::function<void(DataPtr, Dimension)> CreateFuncType;

	/**
	 * Base of NormalCommand and ContextCommand
	 */
	struct Command
	{
		/**
		 * Provided with Opcode name so that we can generate a better error message
		 */
		Command(Opcode op) :
			opName(string(op))
		{ }

		virtual ~Command() {}

		/**
		 * Adapt the registered command into the context
		 * i.e. unpack the context tuple as extra arguments to the command function.
		 * @param context
		 * @return
		 */
		virtual CommandFuncType adapt_context(OpContextBase::Ptr context) = 0;

	protected:
		string opName;
	};
	// WARNING if you TYPEDEF_PTR(Command) inside Command struct, for some reason
	// you have to add typename to 'CommandPtr' everywhere you use Ptr
	// Maybe a nested class problem, idk.
	TYPEDEF_PTR_EXTERNAL(Command);

	/**
	 * Without any context, normal commands are functions with signature CommandFuncType
	 */
	struct NormalCommand : public Command
	{
		/**
		 * Provided with Opcode name so that we can generate a better error message
		 */
		NormalCommand(Opcode op, CommandFuncType cmd_) :
			Command(op), cmd(cmd_)
		{ }

		CommandFuncType adapt_context(OpContextBase::Ptr context)
		{
			LMN_ASSERT_THROW(!context,
				EngineException("OpContext in Instruction is not null: \n\""
					+ Command::opName +"\" "
					"is not a NormalCommand, should be registered as ContextCommand instead."));
			return cmd;
		}

		template<typename ...ArgT>
		static CommandPtr make(ArgT&& ... args)
		{
			return std::static_pointer_cast<Command>(
					std::make_shared<NormalCommand>(
							std::forward<ArgT>(args) ...));
		}
	private:
		CommandFuncType cmd; // Engine<DataT>::CommandFuncType
	};

	/**
	 * Context commands are functors with signature
	 * void(vector<DataPtr>, DataPtr, bool, ExtraContextArgT1, ExtraContextArgT2 ...)
	 */
	// Any type in <...ContextArgT> are extra "environmental context" parameters
	template <typename... ContextArgT>
	struct ContextCommand : public Command
	{
		LMN_STATIC_ASSERT(sizeof...(ContextArgT) > 0,
				"ContextCommand must have at least 1 template ArgType");

		typedef std::function<void(vector<DataPtr>, DataPtr, bool, ContextArgT...)> ContextFuncType;

		/**
		 * Provided with Opcode name so that we can generate a better error message
		 */
		ContextCommand(Opcode op, ContextFuncType contextCmd_):
			Command(op), contextCmd(contextCmd_)
		{}

		CommandFuncType adapt_context(OpContextBase::Ptr context)
		{
			return adapt_context_helper(context,
					typename unpack_gens<sizeof...(ContextArgT)>::type());
		}

		template<typename ...ArgT>
		static CommandPtr make(ArgT&& ... args)
		{
			return std::static_pointer_cast<Command>(
					std::make_shared<ContextCommand<ContextArgT...>>(
							std::forward<ArgT>(args) ...));
		}
	private:
		ContextFuncType contextCmd;

		// helper: unroll std::tuple parameter pack
		template<int ...S>
		CommandFuncType adapt_context_helper(OpContextBase::Ptr contextBase, unpack_seq<S...>)
		{
			LMN_ASSERT_NULLPTR(contextBase,
				EngineException("\""+ Command::opName + "\" is registered as a ContextCommand\n"
						"OpContext in Instruction must be specified (now it's nullptr)"));

			auto context = OpContextBase::cast<ContextArgT...>(contextBase);

			LMN_ASSERT_NULLPTR(context,
				EngineException("OpContext fails to supply the correct number/types of extra context parameters"));

			auto contextArgPack = context->get_context_arg_pack();

			return [=](vector<DataPtr> reads, DataPtr write, bool is_initialized)
			{
				contextCmd(reads, write, is_initialized, std::get<S>(contextArgPack) ...);
			};
		}
	};

	// Call in Engine ctor
	void register_create_op(CreateFuncType createFunc)
	{
		this->command_create = createFunc;
	}

	void register_normal_op(Opcode op, CommandFuncType cmd)
	{
		this->commandMap[op] = NormalCommand::make(op, cmd);
	}

	/**
	 * Register opcode with context
	 * @param op
	 * @param cmd
	 */
	template<typename ... ContextArgT>
	void register_context_op(Opcode op,
			typename ContextCommand<ContextArgT...>::ContextFuncType cmd)
	{
		this->commandMap[op] = ContextCommand<ContextArgT...>::make(op, cmd);
	}

	/**************************************
	********** Compile & execute ***********
	**************************************/
	/**
	 * If the routine has already been compiled,
	 * clear the executable and recompile.
	 */
	virtual void compile(Routine::Ptr routine)
	{
		routine->executables.clear();
		for (Instruction& instr : routine->instructions)
		{
			// FIXME do we need this op?
			if (starts_with(instr.opcode, "create_null"))
				continue;

			vector<DataPtr> reads;
			for (int addr : instr.readAddrs)
				reads.push_back(memoryPool[addr]);

			int writeAddr = instr.writeAddr;

			DataPtr write = memoryPool[writeAddr];

			if (instr.opcode == "create")
			{
				LMN_ASSERT_NULLPTR(instr.context,
					EngineException("context variable (Dimension) is not supplied as part of "
							"the instruction for create command."));

				// Get the context directly
				std::tuple<Dimension> dim =
					OpContextBase::cast<Dimension>(instr.context)->get_context_arg_pack();

				routine->executables.push_back([=]() {
					if (!this->memoryPool.is_initialized(writeAddr))
					{
						this->command_create(write, std::get<0>(dim));
						memoryPool.set_initialized(writeAddr);
					}
				});
			}
			else
			{
				LMN_ASSERT_THROW(key_exists(this->commandMap, instr.opcode),
						EngineException(string("Engine compilation failure: ") +
							"Opcode \"" + string(instr.opcode) + "\" not registered."));

				CommandFuncType cmd = this->commandMap[instr.opcode]->adapt_context(instr.context);
				// value capture by '=' includes 'this'
				routine->executables.push_back([=]() {
					cmd(reads, write, this->memoryPool.is_initialized(writeAddr));
					memoryPool.set_initialized(writeAddr);
				});
			}
		}
	}

protected:
	MemoryPool<DataT> memoryPool;

	/**
	 * Add your "assembly" function addresses for each Opcode
	 */
	std::unordered_map<Opcode, CommandPtr> commandMap;
	CreateFuncType command_create;
};

#endif /* ENGINE_H_ */
