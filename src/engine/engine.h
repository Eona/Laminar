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

	int push(const DataT& data)
	{
		memory.push_back(data);
		return size() - 1;
	}

	int alloc()
	{
		return this->push(DataT());
	}

	DataT& read(int i)
	{
		assert_throw(i < size(),
			EngineException("memory read out of bound."));
		return memory[i];
	}

	DataT& operator[](int i)
	{
		return this->read(i);
	}

	void write(int i, const DataT& data)
	{
		assert_throw(i < size(),
			EngineException("(lvalue) memory write out of bound."));
		memory[i] = data;
	}

	void write(int i, DataT&& data)
	{
		assert_throw(i < size(),
			EngineException("(rvalue) memory write out of bound."));
		memory[i] = data;
	}

	int size()
	{
		return memory.size();
	}

	template<typename T>
	friend ostream& operator<<(ostream& os, MemoryPool<T>& memoryPool);

private:
	vector<DataT> memory;
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
	TensorNode(int _addr) :
		addr(_addr)
	{}

	typedef shared_ptr<TensorNode> Ptr;

	template<typename ...ArgT>
	static TensorNode::Ptr make(ArgT&& ... args)
	{
		return std::make_shared<TensorNode>(
						std::forward<ArgT>(args) ...);
	}

	vector<Ptr> children;
	vector<Ptr> parents;
	int addr; // real memory address in Engine
};

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
