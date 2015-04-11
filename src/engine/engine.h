/*
 * Eona Studio (c) 2015
 */


#ifndef ENGINE_H_
#define ENGINE_H_

#include "../global_utils.h"

template<typename DataT>
class MemoryPool
{
public:
	MemoryPool()
	{ }

	virtual ~MemoryPool() =default;

	using DataType = DataT;

	int alloc()
	{
		memory.push_back(DataT());
		return size() - 1;
	}

	DataT& read(int i)
	{
		assert_throw(i < size(),
			EngineException("memory read out of bound."));
		return memory[i];
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

protected:
	vector<DataT> memory;
};

template<typename T>
ostream& operator<<(ostream& os, MemoryPool<T>& memoryPool)
{
	os << "MemoryPool" << memoryPool.memory;
	return os;
}


class EngineBase
{
public:
	EngineBase()
	{ }

	virtual ~EngineBase() =default;

	typedef shared_ptr<EngineBase> Ptr;

	template<typename EngineBaseT, typename ...ArgT>
	static EngineBase::Ptr make(ArgT&& ... args)
	{
		return static_cast<EngineBase::Ptr>(
				std::make_shared<EngineBaseT>(
						std::forward<ArgT>(args) ...));
	}

	/**
	 * Down cast EngineBasePtr to a specific layer type
	 */
	template<typename EngineBaseT>
	static shared_ptr<EngineBaseT> cast(EngineBase::Ptr layer)
	{
		return std::dynamic_pointer_cast<EngineBaseT>(layer);
	}
};

TypedefPtr(EngineBase);


template<typename DataT>
class Engine : EngineBase
{
public:

protected:
	MemoryPool<DataT> memoryPool;
};



#endif /* ENGINE_H_ */
