/*
 * Eona Studio (c) 2015
 */


#ifndef ENGINE_H_
#define ENGINE_H_

#include "../global_utils.h"
#include "instructions.h"

template<typename DataT>
class MemoryPool
{
static_assert(std::is_default_constructible<DataT>::value,
		"Data type in MemoryPool must be default constructible.");

public:
	MemoryPool()
	{ }

	virtual ~MemoryPool() {};

	using DataType = DataT;

	int alloc()
	{
		this->memory.push_back(DataT());
		this->initialized.push_back(false);
		return size() - 1;
	}

	DataT& operator[](int i)
	{
		assert_throw(i < size(),
			EngineException("memory read out of bound."));
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

	void set_dim(int i, vector<int> dim)
	{
		this->dimensions[i] = dim;
	}

	vector<int>& dim(int i)
	{
		return this->dimensions[i];
	}

	int size()
	{
		return memory.size();
	}

	void reset()
	{
		this->memory.clear();
		this->initialized.clear();
		this->dimensions.clear();
	}

	template<typename T>
	friend ostream& operator<<(ostream& os, MemoryPool<T>& memoryPool);

private:
	// All the following should have the same size
	vector<DataT> memory;
	// test if things are default initialized.
	vector<bool> initialized;
	// dimension of initialized tensor at memory addr
	std::unordered_map<int, vector<int> > dimensions;
};

template<typename T>
ostream& operator<<(ostream& os, MemoryPool<T>& memoryPool)
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

	typedef shared_ptr<TensorNode> Ptr;

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

ostream& operator<<(ostream& os, TensorNode& node)
{
	os << string(node);
	return os;
}

class EngineBase
{
public:
	EngineBase()
	{ }

	virtual ~EngineBase() {};

	void upload(Instruction instr)
	{
		instructions.push_back(instr);
	}

	virtual int alloc() = 0;

	virtual void set_dim(int addr, vector<int> dim) = 0;

	virtual void reset() = 0;

	/**
	 * Construct a DAG of data dependencies
	 */
	virtual void construct_graph()
	{
		for (Instruction& instr : this->instructions)
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

	virtual void print_instructions()
	{
		for (auto& instr : this->instructions)
			cout << instr << "\n";
	}

	/************************************/
	typedef shared_ptr<EngineBase> Ptr;

	template<typename EngineT, typename ...ArgT>
	static shared_ptr<EngineT> make(ArgT&& ... args)
	{
		static_assert(std::is_base_of<EngineBase, EngineT>::value,
				"Engine type parameter must be a subclass of EngineBase");
		return std::make_shared<EngineT>(
						std::forward<ArgT>(args) ...);
	}

	/**
	 * Downcast
	 */
	template<typename EngineBaseT>
	static shared_ptr<EngineBaseT> cast(EngineBase::Ptr engine)
	{
		return std::dynamic_pointer_cast<EngineBaseT>(engine);
	}

protected:
	vector<Instruction> instructions;
	/**
	 * Only contains TensorNodes created by "create" opcode.
	 */
	vector<TensorNode::Ptr> createdNodes;
};

TYPEDEF_PTR(EngineBase);


template<typename DataT>
class Engine : public EngineBase
{
public:
	Engine() :
		EngineBase()
	{ }

	virtual ~Engine() {};

	virtual int alloc()
	{
		return memoryPool.alloc();
	}

	virtual void set_dim(int addr, vector<int> dim)
	{
		memoryPool.set_dim(addr, dim);
	}

	virtual void reset()
	{
		this->memoryPool.reset();
	}

	/*********** Register "assembly" implementation ***********/
	// (readAddrs, writeAddr, is_initialized)
	typedef std::function<void(vector<DataT*>, DataT*, bool)> OpcodeFuncType;
	// specifically for Opcode create_dim(writeAddr, dim)
	typedef std::function<void(DataT*, vector<int>)> CreateFuncType;

	// Call in Engine ctor
	void register_create(CreateFuncType createFunc)
	{
		this->assembly_create = createFunc;
	}

	// Call in Engine ctor
	void register_opcode(Opcode op, OpcodeFuncType opFunc)
	{
		this->assembly_map[op] = opFunc;
	}

	/*********** Compilation and execution ***********/
	vector<std::function<void()>> compile()
	{
		vector<std::function<void()>> assembly;

		for (Instruction& instr : this->instructions)
		{
			// TODO do we need this op?
			if (starts_with(instr.opcode, "create_null"))
				continue;

			vector<DataT*> reads;
			for (int addr : instr.readAddrs)
				reads.push_back(&memoryPool[addr]);

			int writeAddr = instr.writeAddr;

			DataT *write = &memoryPool[writeAddr];

			if (instr.opcode == "create")
			{
				vector<int> dim = memoryPool.dim(writeAddr);
				CreateFuncType assembly_create = this->assembly_create;
				assembly.push_back([=]() {
					if (!this->memoryPool.is_initialized(writeAddr))
					{
						assembly_create(write, dim);
						memoryPool.set_initialized(writeAddr);
					}
				});
			}
			else
			{
				if (!key_exists(this->assembly_map, instr.opcode))
					throw EngineException(string("Engine compilation failure: ") +
							"Opcode \"" + string(instr.opcode) + "\" not registered.");

				OpcodeFuncType assembly_op = this->assembly_map[instr.opcode];
				// value capture by '=' includes 'this'
				assembly.push_back([=]() {
					assembly_op(reads, write, this->memoryPool.is_initialized(writeAddr));
					memoryPool.set_initialized(writeAddr);
				});
			}
		}

		return assembly;
	}

	void execute()
	{
		for (auto assembly : this->compile())
			assembly();
	}

protected:
	MemoryPool<DataT> memoryPool;

	/**
	 * Add your "assembly" function addresses for each Opcode
	 */
	std::unordered_map<Opcode, OpcodeFuncType> assembly_map;
	CreateFuncType assembly_create;
};

#endif /* ENGINE_H_ */
