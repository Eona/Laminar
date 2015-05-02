#Design Document
##Error Handling

Laminar framework has a hierarchy of exceptions. A function will thrown an exception of appropriate type if it fails the consistency check. 

###Laminar Exception
The top level exception. `LaminarException` directly extends `std::exception` and is the highest level exception type in the library. 

It is rarely thrown by itself, except when the error is too hard to classify by more detailed derived exceptions. 

###Network Exception
A `NetworkException` is thrown by the `Network` base class, or any of its derived classes `ForwardNetwork` and `RecurrentNetwork`.

The exception is typically thrown when there are inconsistencies in the network topology. 

For example, the following situations will trigger a `NetworkException`

- A non-recurrent connection attempts to connect a layer to itself. <br><br>
- The network has no input layer.<br><br>
- The network has no loss layer (the last layer must be a derived class of `LossLayer`)<br><br>
- For recurrent networks, an error occurs when a connection's temporal skip value exceeds the global `maxTemporalSkip`. <br><br>

###Component Exception
A `ComponentException` is thrown when either `Layer` or `Connection` sees an inconsistency. 

For example, the following situations will trigger `ComponentException`

- `Layer` receives an invalid `Dimension`
- `ConstantConnection` attempts to connect two layers with different dimensions. 
- `GatedConnection` dimension mismatch.

###Engine Exception
An `EngineException` is thrown whenever an internal inconsistency occurs in either the virtual engine itself or any of its backends. 

###Tensor Exception
A `TensorException` is thrown when invalid operations are performed on a tensor

###Learning Exception

A `LearningException` is thrown when the learning procedure fails or sees a logical inconsistency.

Any of the ***O-E-S-S-E-O*** modules (Optimizer, Evaluator, Stop criteria, Serializer, Evaluation schedule, Observer) can throw a `LearningException`

###Data Exception

A `DataException` is normally thrown by the `DataManager`. 

Any invalid IO operations that read/load training/testing data will trigger a `DataException`. 

When the training/testing data streams are depleted but the user fails to reset a new epoch, `DataException` will also be triggered. 

#Interface Design
![Interface](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/overview.png)

At the highest level, the _Laminar_ architecture can be roughly divided into three parts:

## Network Topology
You can build any imaginable neural network topology as long as you can define a computational graph ("network" is used interchangeably here). 
	 
The network is an abstract container that encapsulates the nodes ("layers") and edges ("connections"). It does not do actual computation. Instead, it uploads instructions to the virtual engine. 
	 
The network is analogous to a high-level language. 

###Virtual Engine

The virtual engine receives instructions from the network. The instructions encode when to allocate memory and what computation needs to be performed on which memory address. All the memory are managed by an internal object pool. 
	 
In a sense, the virtual engine is an intermediate agent. It is somewhat analogous to a compiler that translates higher-level instructions to lower-level representations. It delegates computation to the next module.
	 
##Actual computation backends
	
This is the main workhorse that powers the deep learning process. *Laminar*'s speed is limited only by the backend it runs on. All computation backends conform to a unified interface that integrates with the virtual engine. In this way, switching between drastically different computation hardware becomes very easy. We currently have the following backends choices:

- Eigen (CPU): based on the [Eigen](eigen.tuxfamily.org) library, a template header library for high-performance linear algebra computation. Eigen has transparent multithreading support. <br><br>
- Vecmat (CPU): the C++ `std::vector`-based backend. This is implemented from scratch and is predictably the slowest of all backends. It has been developed mainly for debugging purposes and should not be used in deployment. <br><br>
- CUDA: a plain CUDA backend implementation using [NVidia's toolkit](http://www.nvidia.com/object/cuda_home_new.html). We have implemented custom GPU kernels for matrix operations, optimized by shared memory. <br><br>
- OpenCL: counts as two backends, because OpenCL can run on both CPU and GPU. <br><br>
	- In CPU mode, intel-based OpenCL invokes [ThreadBuildingBlock](https://www.threadingbuildingblocks.org/) under the hood, a high-performance multithreading platform. <br><br>
	- In GPU mode, OpenCL can run on both NVidia and non-NVidia graphics processors. For example, OpenCL runs on _Intel Iris Pro_ GPU which does not support CUDA. _Intel Iris Pro_ is commonly found on Macbook Pros.<br><br>
- cuBLAS: a high-performance GPU [linear algebra library](https://developer.nvidia.com/cuBLAS). It is included natively as part of the CUDA toolkit. From our experiments, cuBLAS is the fastest compute engine of all six. 

#Optimizations

##Network Optimization

The temporal skip implementation has certain optimizations

	net.init_max_temporal_skip(3);

If you set the temporal skip value to a special `Layer::UNLIMITED_TEMPORAL_SKIP` constant, the network will save the full gradient history so that you can jump to an arbitrary frame back in time. 

It is recommended that you keep the value as low as possible, because the more temporal skip you set, the more gradient history the network will save. This might result in unnecessarily large memory footprint. `Layer::UNLIMITED_TEMPORAL_SKIP` is typically used for learning debugging, if you wish to inspect the gradient history at every time frame. 

The very first implementation did not have this option - all gradient history is saved in a huge container even though only a tiny fraction of it is active at a time. 

In a major revision, we optimize the network such that a "gradient window" slides back through history. The gradient history is saved up to `maxTemporalSkip`. This drastically decreases memory footprint and increases efficiency. 

##Engine Optimization

###Temporary variable elimination
We have implemented special optimizers that scan the sequence of instruction queue and eliminate unnecessary temporary variables. 

###Math operation optimization

The following line

	Tensor t1, t2, t3;
	t1 = lmn::transpose(t2) * t3;

is translated to the following list of instructions:

    transpose t2 -> tmp
    mult_t_t tmp t3 -> tmp2
    assign tmp2 -> t1

Our optimizer is able to squash the above into a single instruction

    transpose_mult t2, t3 -> t1

Note that a `transpose_mult` is a single instruction on BLAS. It is much more efficient than `transpose()` first and then multiply the two matrices. 

#v1.2 Future Release
##Convolutional Neural Network

In v1.2, we would like to incorporate convolutional NNs into the *Laminar* framework. 

Convolutional Neural Networks (CNN) are biologically-inspired variants of MLPs. From Hubel and Wiesel’s early work on the cat’s visual cortex, we know the visual cortex contains a complex arrangement of cells. 

These cells are sensitive to small sub-regions of the visual field, called a receptive field. The sub-regions are tiled to cover the entire visual field. These cells act as local filters over the input space and are well-suited to exploit the strong spatially local correlation present in natural images.

Additionally, two basic cell types have been identified: Simple cells respond maximally to specific edge-like patterns within their receptive field. Complex cells have larger receptive fields and are locally invariant to the exact position of the pattern.

CNNs exploit spatially-local correlation by enforcing a local connectivity pattern between neurons of adjacent layers. In other words, the inputs of hidden units in layer *m* are from a subset of units in layer *m - 1*, units that have spatially contiguous receptive fields. 

In addition, CNNs feature *shared weights* that drastically decrease the number of parameters to be trained, thus boosting learning efficiency dramatically. 

##Neural Turing Machine
Neural Turing Machines are novel architectures that extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes. 

The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-to-end, allowing it to be efficiently trained with gradient descent. 

Preliminary results in cutting edge research demonstrate that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples.

##Fortran backend
In addition to the 6 backends shipped with the current *Laminar* release, we would like to have a high-performance Fortran backend that powers the neural networks. 

##TBB backend

Intel *ThreadBuildingBlock* (TBB) is a mature library that uses multithreading to scale up CPU clusters. 

We would like *Laminar* to run on large-scale cloud clusters with TBB. 