/*
 * Eona Studio (c) 2015
 */

#ifndef DATA_MANAGER_H_
#define DATA_MANAGER_H_

#include "../global_utils.h"

class DataManagerBase
{
public:
	virtual ~DataManagerBase() {}

	/************************************/
	typedef shared_ptr<DataManagerBase> Ptr;

	template<typename ManagerT, typename ...ArgT>
	static shared_ptr<ManagerT> make(ArgT&& ... args)
	{
		return std::make_shared<ManagerT>(
						std::forward<ArgT>(args) ...);
	}

	/**
	 * Downcast
	 */
	template<typename ManagerT>
	static shared_ptr<ManagerT> cast(DataManagerBase::Ptr manager)
	{
		return std::dynamic_pointer_cast<ManagerT>(manager);
	}
};

template<typename DataT>
class DataManager : public DataManagerBase
{
public:
	virtual void fill_input(DataT *write) = 0;

	virtual void fill_target(DataT *write) = 0;

};

#endif /* DATA_MANAGER_H_ */