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

	/**
	 * Dimension of the tensor
	 */
	int alloc(vector<int> dim)
	{
		this->memory.push_back(DataT());
		this->initialized.push_back(false);
		this->dimensions.push_back(dim);
		return size() - 1;
	}

	DataT* memory_ptr(int i)
	{
		assert_throw(i < size(),
			EngineException("memory pointer request out of bound."));
		return &memory[i];
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

	vector<int> dim(int i)
	{
		return this->dimensions[i];
	}

	int size()
	{
		return memory.size();
	}

	template<typename T>
	friend ostream& operator<<(ostream& os, MemoryPool<T>& memoryPool);

private:
	// All the following should have the same size
	vector<DataT> memory;
	// test if things are default initialized.
	vector<bool> initialized;
	// dimension of each tensor
	vector<vector<int> > dimensions;
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

	// Requires knowledge of the memory pool
	virtual int alloc() = 0;

	/**
	 * Construct a DAG of data dependencies
	 */
	virtual void construct_graph()
	{
		for (Instruction& instr : this->instructions)
		{
			vector<int>& reads = instr.readAddrs;
			int write = instr.writeAddr;
			string op = string(instr.code);
			if(op == "create")
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
			if (instr.code == "destroy")
			{
				auto instr_1 = i[-1]; // *(i - 1)
				auto instr_2 = i[-2];
				// instr_2 { t+t: [2, 2] -> 4 }
				// instr_1 { copy: [4] -> 3 }
				// instr { destroy: [] -> 4 }
				// optimize and eliminate temporary '4'
				// instr_new { t+t: [2, 2] -> 3 }
				if (instr_1.code == "copy"
					&& instr_1.readAddrs[0] == instr.writeAddr
					&& instr_2.writeAddr == instr_1.readAddrs[0])
				{
					// instr_3 might have { create: [] -> 4 }
					auto instr_3 = i[-3];
					if (instr_3.code == "create"
						&& instr_3.writeAddr == instr.writeAddr)
					{
						// eliminate all 4 instructions instr_3 ... instr, inclusive
						i = instructions.erase(i - 3, i + 1);
					}
					else
						i = instructions.erase(i - 2, i + 1);
					// Add new combined instr
					instructions.insert(i,
						Instruction(instr_2.code, instr_2.readAddrs, instr_1.writeAddr));
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
		return std::make_shared<EngineT>(
						std::forward<ArgT>(args) ...);
	}

	/**
	 * Down cast EngineBasePtr to a specific layer type
	 */
	template<typename EngineBaseT>
	static shared_ptr<EngineBaseT> cast(EngineBase::Ptr layer)
	{
		return std::dynamic_pointer_cast<EngineBaseT>(layer);
	}

protected:
	vector<Instruction> instructions;
	/**
	 * Only contains TensorNodes created by "create" opcode.
	 */
	vector<TensorNode::Ptr> createdNodes;
};

TypedefPtr(EngineBase);


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

protected:
	MemoryPool<DataT> memoryPool;
};

#endif /* ENGINE_H_ */
