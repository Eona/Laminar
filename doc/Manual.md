<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
#*Laminar* Manual

- [*Laminar* Architecture](#laminar-architecture)
- [Network Topology](#network-topology)
  - [Network](#network)
    - [Layer](#layer)
    - [Connection](#connection)
    - [Bias](#bias)
    - [Composite](#composite)
  - [Recurrent Network](#recurrent-network)
    - [Temporal skip](#temporal-skip)
    - [Recurrent connection](#recurrent-connection)
    - [Temporal skip connection](#temporal-skip-connection)
- [Component hierarchy](#component-hierarchy)
  - [Component](#component)
  - [Layer (base class)](#layer-base-class)
    - [ConstantLayer](#constantlayer)
    - [Activation Layer](#activation-layer)
      - [SigmoidLayer](#sigmoidlayer)
      - [TanhLayer](#tanhlayer)
      - [CosineLayer](#cosinelayer)
      - [ScalarLayer](#scalarlayer)
      - [DebugLayer](#debuglayer)
    - [Loss Layer](#loss-layer)
      - [SquareLossLayer](#squarelosslayer)
      - [LabelSoftmaxEntropyLayer](#labelsoftmaxentropylayer)
  - [Connection (base class)](#connection-base-class)
    - [ConstantConnection](#constantconnection)
    - [FullConnection](#fullconnection)
    - [GatedConnection](#gatedconnection)
      - [GatedSigmoidConnection](#gatedsigmoidconnection)
      - [GatedTanhConnection](#gatedtanhconnection)
- [Virtual Engine](#virtual-engine)
  - [TensorBase Hierarchy](#tensorbase-hierarchy)
    - [Big-Six operators overloaded](#big-six-operators-overloaded)
    - [Standalone math operators overloaded](#standalone-math-operators-overloaded)
    - [Member math operators overloaded](#member-math-operators-overloaded)
    - [Special functions in lmn:: namespace](#special-functions-in-lmn-namespace)
  - [Instruction](#instruction)
    - [Opcode](#opcode)
    - [Read addresses](#read-addresses)
    - [Write address](#write-address)
    - [OpContext](#opcontext)
- [Computation Backend](#computation-backend)
  - [Built-in backends](#built-in-backends)
    - [CUDA](#cuda)
    - [cuBLAS](#cublas)
    - [OpenCL](#opencl)
    - [Eigen](#eigen)
    - [Vecmat](#vecmat)
  - [Rolling Your Own Backend](#rolling-your-own-backend)
    - [Specify a `DataType`](#specify-a-datatype)
    - [Implement functions associated with each *Laminar* opcode](#implement-functions-associated-with-each-laminar-opcode)
    - [Listing of *Laminar* opcodes](#listing-of-laminar-opcodes)
    - [Register your functions with the opcodes](#register-your-functions-with-the-opcodes)
    - [Add your own opcodes](#add-your-own-opcodes)
- [Learning Modules](#learning-modules)
  - [LearningSession](#learningsession)
  - [LearningState](#learningstate)
  - [Optimizer](#optimizer)
    - [SGD](#sgd)
    - [MomentumGD](#momentumgd)
    - [ClippedMomentum](#clippedmomentum)
    - [NesterovMomentum](#nesterovmomentum)
    - [Adagrad](#adagrad)
    - [RMSprop](#rmsprop)
  - [Evaluator](#evaluator)
  - [StopCriteria](#stopcriteria)
    - [MaxEpochStopper](#maxepochstopper)
  - [Serializer](#serializer)
    - [NullSerializer](#nullserializer)
  - [EvalSchedule](#evalschedule)
    - [NullSchedule](#nullschedule)
    - [NoValidationSchedule](#novalidationschedule)
    - [NoTestingSchedule](#notestingschedule)
    - [EpochIntervalSchedule](#epochintervalschedule)
  - [Observer](#observer)
    - [NullObserver](#nullobserver)
    - [MinibatchObserver](#minibatchobserver)
- [Exception](#exception)
  - [Laminar Exception](#laminar-exception)
  - [Network Exception](#network-exception)
  - [Component Exception](#component-exception)
  - [Engine Exception](#engine-exception)
  - [Tensor Exception](#tensor-exception)
  - [Learning Exception](#learning-exception)
  - [Data Exception](#data-exception)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

#*Laminar* Architecture

The manual is organized in the order of *Laminar*'s high-level architecture. 

![Big picture](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/Laminar_Big_Picture.png)

#Network Topology

##Network
*Laminar* framework currently supports two major networks. `ForwardNet` and `RecurrentNet`, both extends the `Network` super class. 

`ForwardNet` has the following methods to build an arbitrary topology:

###Layer

	add_layer(Layer::Ptr)

Adds an existing layer pointer to the network architecture. 

###Connection
	add_connection(Connection::Ptr)

Adds an existing connection to the network topology. 

	new_connection<ConnectionT>(Layer::Ptr in, Layer::Ptr out)

Constructs a new connection internally, and is a convenience method that's equivalent to 

    auto conn = Connection::make<ConnectionT>(in, out);
    net->add_connection(conn);

###Bias

	new_bias_layer(Layer::Ptr)`

Constructs a new bias layer and adds it to the layer being passed in as a parameter.

The bias layer will have exactly the same dimension as the input layer. 

It is equivalent to 

    auto bias = Layer::make<BiasLayer>(inLayer->dim());
    net->add_connection(bias, inLayer);
    net->add_layer(bias);

###Composite
There are two flavors of `add_composite()`:

    template<typename CompositeT>
	void add_composite(std::shared_ptr<CompositeT> composite);

and 

    template<typename CompositeT>
	void add_composite(CompositeT& composite);

You can either create a `shared_ptr` of `Composite` or an object of `Composite` and pass in a reference. 

The method adds a `Composite` that automates subgraph generation for the network.

The most notable use case of `Composite` is `LstmComposite`. LSTM networks have extremely complicated topology that involves almost a dozen layers, `FullConnection` and `GatedTahnConnection`. 

While you can always construct an LSTM from scratch, the `Composite` class greatly improves your productivity. 

##Recurrent Network

Due to the temporal dimension absent from forward networks, a `RecurrentNetwork` has a few additional methods:

###Temporal skip

	init_max_temporal_skip(temporalSkipValue)

Sets the maximal temporal skip value.

If you set the temporal skip value to a special `Layer::UNLIMITED_TEMPORAL_SKIP` constant, the network will save the full gradient history so that you can jump to an arbitrary frame back in time. 

It is recommended that you keep the value as low as possible, because the more temporal skip you set, the more gradient history the network will save. This might result in unnecessarily large memory footprint. `Layer::UNLIMITED_TEMPORAL_SKIP` is typically used for learning debugging, if you wish to inspect the gradient history at every time frame. 

###Recurrent connection

	add_recur_connection(Connection::Ptr)

Adds an existing recurrent connection that has default temporal skip value = 1. 

	new_recur_connection<ConnectionT>(Layer::Ptr in, Layer::Ptr out)

Constructs a new recurrent connection that connections the *t - 1* frame of `inLayer` to the current frame *t* of `outLayer`. 

The method is equivalent to 

    auto conn = Connection::make<ConnectionT>(inLayer, outLayer);
    net->add_recur_connection(conn);

###Temporal skip connection
Temporal skip connections are a generalization of traditional 1-frame recurrent connections. 

`add_recur_skip_connection(int temporalSkip, Connection::Ptr)` 

Adds an existing recurrent connection of a specific temporal skip value. 
E.g. if the value is 5, the `inLayer`'s *t - 5* frame value will be forwarded to `outLayer`'s current *t* frame. 

`new_recur_skip_connection<ConnectionT>(int temporalSkip, Layer::Ptr in, Layer::Ptr out)`

Constructs a new recurrent connection of a specific temporal skip value. 

This method is equivalent to

    auto conn = Connection::make<ConnectionT>(inLayer, outLayer);
    net->add_recur_skip_connection(temporalSkip, conn);

To add a temporal skip = 2 connection, 

    net->new_recur_skip_connection<FullConnection>(2, hidden1, hidden1);
    net->new_recur_skip_connection<FullConnection>(3, hidden2, hidden2);


#Component hierarchy

![Components](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/components.png)

##Component
`Component` is the global base class in the hierarchy of network topology. 

It has relatively few methods, most of which are pure virtual methods. 

The most important ones are

`void forward(int inFrame, int outFrame)`

`void backward(int inFrame, int outFrame)`

They encapsulate forward and backward propagation logic, the key to gradient-based deep learning algorithms. 

The actual logic should be implemented by `Layer` and `Connection`.

##Layer (base class)

A layer in *Laminar* terminology is equivalent to a *node* in graph theory. 

All layers extend the abstract super class `Layer`, which has the following concrete methods:

`init_engine(EngineBase::Ptr)`

called internally by `Network` to upload `Tensor` instructions to the virtual engine. 

`init_max_temporal_skip(int temporalSkipValue)`

called internally by `RecurrentNetwork`

`Tensor& in_value(int frame)`
`Tensor& in_gradient(int frame)`
`Tensor& out_value(int frame)`
`Tensor& out_gradient(int frame)`

Are getter methods that return a reference to the internal `Tensor` at time frame `t`. 

There are another set of similar methods that return `Tensor::Ptr` instead

`Tensor::Ptr in_value_ptr(int frame)`
`Tensor::Ptr in_gradient_ptr(int frame)`
`Tensor::Ptr out_value_ptr(int frame)`
`Tensor::Ptr out_gradient_ptr(int frame)`

The `Layer` class also defines the interface for the following pure virtual methods

	virtual void forward_impl(
		Tensor& inValue, 
		Tensor& outValue) = 0;
	
	virtual void backward_impl(
		Tensor& outValue, 
		Tensor& outGradient, 
		Tensor& inValue, 
		Tensor& inGradient) = 0;

The `forward_impl` and `backward_impl` are the actual forward/backward propagation logic. All subclasses of `Layer` implement this interface. 

This is part of the Developer API. If you wish to add `ConvolutionLayer`, for example, you will have to implement `forward_impl` and `backward_impl` logic. 

Each `Layer` (except for the special `BiasLayer`) takes at least one constructor argument: `Dimension dim`, with `Dimension` as a type alias for `vector<int>` to describe the dimensionality of tensors. 

Here is an incomplete figure of the built-in `Layer` hierarchy:

![Layer](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/Layer.png)

###ConstantLayer

`ConstantLayer` is a data-holding layer that does not do any computation. Its `forward_impl` and `backward_impl` do basically nothing. 

What is more, it overrides the getter methods:

`Tensor& out_value(int frame)`
`Tensor& out_gradient(int frame)`
`Tensor::Ptr out_value_ptr(int frame)`
`Tensor::Ptr out_gradient_ptr(int frame)`

such that `out_value` redirects to `in_value` and `out_gradient` redirects to `in_gradient`. This approach saves unnecessary copying and memory footprint.

###Activation Layer

`ActivationLayer` is a derived class from `Layer`. All activation layers implement the following pure virtual methods:

	virtual void forward_impl(
		Tensor& inValue, 
		Tensor& outValue) = 0;
	
	virtual Tensor activation_gradient(
		Tensor& outValue, 
		Tensor& inValue) = 0;

`forward_impl` computes the nonlinear activation function. It is the same pure virtual method directly inherited from `Layer`. 

Instead of implementing `backward_impl`, you will implement `activation_gradient` that computes the gradient for the activation function, if you are to extend the *Laminar* `ActivationLayer` hierarchy. 

All the built-in `ActivationLayer`s are explained below. 

####SigmoidLayer

 Computes the sigmoid function
	
	f(x) = 1 / (1 + exp(-x))

which is the most commonly used non-linear activation function. In fact, sigmoid is the first activation function ever mentioned at the birth of neural networks (called the *perceptron* algorithm by then). 

####TanhLayer

Computes the tanh function

`f(x) = tanh(x) == (exp(2*x) - 1) / (exp(2*x) + 1)`

This is the second most commonly used non-linear activation function and the default function used in `GatedConnection`. It is the nonlinear gate choice for LSTM networks, because it ranges from `-1.0` to `+1.0` and transitions smoothly. 

####CosineLayer

Computes the cosine function

    f(x) = cos(x)

Can be used in certain Fourier-transformation based networks like the *Rahimi-Recht random projection*.

####ScalarLayer

Computes a simple linear scaling function

`f(x) = x * scalar`

####DebugLayer

For debugging only.

###Loss Layer

All derived classes from `LossLayer`  implement the following pure virtual methods:

    virtual Scalar loss_forward_impl(Tensor& inValue, Tensor& targetValue) = 0;

	virtual void loss_backward_impl(Tensor& inValue, Tensor& targetValue, Tensor& inGradient) = 0;

You should implement `loss_forward_impl` and `loss_backward_impl` instead of `forward_impl` and `backward_impl` if you are to add your own loss function. 

Two built-in `LossLayer`s are shipped with the library.  

####SquareLossLayer

Computes the square loss function

	loss(x, y) = (x - y) ** 2

Square loss is typically used for regression tasks. 

####LabelSoftmaxEntropyLayer

Computes the cross-entropy loss function at the ground-truth label

	loss(x, y) = 
	-log(softmax(x) * I[y == ground_truth_label])

`softmax()` is a function that normalizes a prediction vector to be a proper probability distribution. 

Cross-entropy loss is typically used for classification task with one-hot encoding. 

##Connection (base class)

The figure below shows an incomplete picture of the `Connection` hierarchy:

![Connection](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/Connection.png)

All derived classes from `Connection` base class implements the following pure virtual methods:

    virtual void forward_impl(
		Tensor& inlayerOutval, Tensor& outlayerInval) = 0;

	virtual void backward_impl(
		Tensor& outlayerIngrad, Tensor& inlayerOutval, Tensor& inlayerOutgrad) = 0;

They encapsulate the forward/backward propagation logic for the neural network. 

###ConstantConnection

`ConstantConnection` does nothing but forwarding the information as-is to its destination layer. 

`ConstantConnection` requires that its input layer and output layer have the same dimension - because the information is simply copied and forwarded. 

###FullConnection

`FullConnection` is the most commonly used connection in *Laminar*. It contains learnable parameter tensors in a fully-connected fashion. 

`FullConnection` is not meant to be derived from. It should be used as-is. 

###GatedConnection

`GatedConnection` is used extensively in recurrent neural networks as regulators for temporal information flow. 

All derived classes from `GatedConnection` should implement the virtual methods:

    virtual void gated_forward_impl(
	    Tensor& inlayerOutval, 
	    Tensor& gateOutval, 
	    Tensor& outlayerInval);

	virtual void gated_backward_impl(
		Tensor& outlayerIngrad, 
		Tensor& inlayerOutval, 
		Tensor& gateOutval,
		Tensor& inlayerOutgrad, 
		Tensor& gateOutgrad);

There are currently two derived classes. You can implement more as you need.

####GatedSigmoidConnection

A gate regulated by the sigmoid function. This gate will have range `[0, 1]` and corresponding properties.

####GatedTanhConnection

A gate regulated by the tanh function.  This gate will have range `[-1, 1]` and corresponding properties.

#Virtual Engine

##TensorBase Hierarchy

`TensorBase` is the main communication channel between the network topology and the virtual engine. The network uploads instructions (topic of the next subsection) to the virtual engine via `TensorBase`. 

<img src ="https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/tensor.png" width = "330" />

`Tensor` is just a fancy name for multi-dimensional arrays. With respect to most forward and recurrent networks, `Tensor` is synonymous with matrix. 

We make the distinction between `Tensor` and `Scalar`, because certain operations have overloaded meaning for `Tensor` and `Scalar`. For example, `Tensor`-`Tensor` multiplication is of course not the same as `Tensor`-`Scalar` multiplication (which is essentially element-wise scaling). 

###Big-Six operators overloaded

All of the following operators are carefully overloaded for `Tensor` and `Scalar`:

1. Constructor --- uploads `create` instruction
2. Destructor --- `destroy` instruction
3. Copy assignment --- `assign` instruction
4. Copy constructor --- `create` instruction
5. Move assignment --- `assign` instruction
6. Move assignment --- `create` instruction

###Standalone math operators overloaded

Common math operators are overloaded for both `Tensor` and `Scalar`:

- `+`  plus operator
- `-`  minus operator
- `-`  negate operator
- `*`  multiplication operator

###Member math operators overloaded

Math operators that are member functions are also overloaded:

- `+=`  plus-assign operator
- `-=`  minus-assign operator
- `*=`  multiply-assign operator

###Special functions in lmn:: namespace

Functions like `lmn::sigmoid(const Tensor&)` will upload an instruction with opcode `"sigmoid"` to the virtual engine. 

You will see a complete listing in the next section. 

##Instruction
`Instruction` is a class that constitutes an "intermediate representation" between high-level math logic and low-level computation. 

Each `Instruction` has four fields:

###Opcode

`Opcode` is a wrapper around `std::string` and implements `std::hash<Opcode>` interface. 

*Laminar* has around 20 built-in opcodes. If you want to implement your own backend, you will have to register the opcodes with your backend's actual implementation. 

Note that you don't have to register all the opcodes though. For example, if you do not plan to use `ClippedMomentum` optimizer, you don't have to implement the `"clip"` opcode. 

###Read addresses

Read address is a field in `Instruction`, with the type `std::vector<int>`. Note that each memory address is an integer that refers to a location in the internal memory pool. 

Reading can have multiple addresses. For example, adding two tensors requires two operands, so the read address will have length 2.

###Write address
An int represents the write address for an instruction, which always has one and only one write address as for the current release. 

###OpContext

There are two types of opcodes:

1. Normal opcodes with no operational context
2. Context opcodes

We introduce the concept of "context opcode" because an instruction might need extra information (other than tensors) to perform an operation. For example, the opcode `"perturb"`, an essential instruction used in gradient checking, requires not only a tensor address to be operated on, but also a `DimIndex` that specifies which entry to be perturbed and a constant amount of perturbation (typically `1e-6f`, a very small positive number). The extra `DimIndex` and `float` information are what we call "operation context variables". 

`OpContext` class is designed to work with *any* number of context variables of *any* type. 

An example `Instruction` is shown below:

    Instruction instr(
	    // opcode
	    Opcode("my_op",
	    // read addresses
	    { 1, 2 },
	    // write address
	    3,
	    // operation context with 3 variables
	    OpContext<int, string, std::function<float(float)>>(
	    3, // int context
	    "this is a context", // string context
	    [](float x) { return x * x; } // std::function context
	    )
	 );

As you can see from the above example, an instruction can take arbitrary number of arbitrary heterogenous types.


#Computation Backend

##Built-in backends

There are 6 built-in backends shipped with *Laminar* library. Using them is as easy as changing a single line of code.

###CUDA

    EngineBase::make<CudaEngine>();

CUDA stands for Compute Unified Device Architecture, is a parallel computing platform and programming model created by NVIDIA and implemented by the graphics processing units (GPUs) that they produce. It is a simplistic interface for NVIDIA Graphical Card. At the same time, it provides extensive access to many low-level functions such as event-based profiling and memory profiling.

###cuBLAS

	EngineBase::make<CublasEngine>(); // can take an optional memory profiler arg

cuBLAS is a linear algebra library of CUDA architecture. cuBLAS is essentially a GPU-accelerated version of the complete standard BLAS library that delivers 6x to 17x faster performance than the latest MKL BLAS.

###OpenCL

	EngineBase::make<OpenclEngine>(); 

OpenCL is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors. Although OpenCL is not as easy to use as CUDA in terms of interface, it provides the possibility of our framework running on numerous devices other than computers that have standard Graphics Processing Unit.

Our OpenCL backend counts as two, because it can run in either CPU mode or GPU mode. 

###Eigen

    EngineBase::make<EigenEngine>();

Eigen is a high-level C++ library of template headers for linear algebra, matrix and vector operations, numerical solvers and related algorithms. It is heavily optimized, but can only run on a single thread if not configured with multi-processing option.

###Vecmat

    EngineBase::make<VecmatEngine>();

`Vecmat` stands for vector matrix, which is implemented using std::vector library. We did not implement any optimization algorithms in this framework so it was expected to be the most inefficient backend. It serves as a baseline of our performance profiling.

##Rolling Your Own Backend

A complete Developer API is provided for rolling your own computational backend. 

While built-in support for CPUs and GPUs seem enough for most research tasks, you might want to run *Laminar* on a quantum computer or Ironman's suit in the future. 

Developer API allows you to implement your custom backend in several clean steps. 

###Specify a `DataType`

First you need to specify a data type for your quantum tensor. 

    class MyEngine : Engine<MyDataType>

A `Scalar` can be treated as a 1-by-1 `Tensor`

###Implement functions associated with each *Laminar* opcode

All normal (i.e. without any context) operations should have the following signature:

    void op(
	    vector<std::shared_ptr<QuantumTensor>> reads, 
	    std::shared_ptr<QuantumTensor> write, 
	    bool is_initialized);

Your code needs to check for `is_initialized`. If it is false, that means the `write` tensor has not been initialized - you will have to construct it with the right dimension your function requires.

For example, the opcode `mult_t_t` (multiply two tensors) is implemented in Eigen backend as following. `MatrixXf` is Eigen's tensor type. `MatrixXfPtr` is a shared pointer typedef. 

    void mult_t_t(
	    vector<MatrixXfPtr> reads, 
	    MatrixXfPtr write, 
	    bool is_initialized)
	{
		MatrixXf& in1 = *reads[0];
		MatrixXf& in2 = *reads[1];
		
		if (!is_initialized)
			*write = MatrixXf::zero(
					in1.rows(), in2.cols());

		*write = in1 * in2;
	}

All context operation functions will have the above 3 arguments plus a number of trailing arguments that convey context information. 

For example, the `perturb` opcode implementation should have the following signature:

    void quantum_perturb(
	    vector<std::shared_ptr<QuantumTensor>> reads, 
	    std::shared_ptr<QuantumTensor> write, 
	    bool is_initialized,
	    DimIndex perturb_index,
	    float perturb_amount);

###Listing of *Laminar* opcodes

- `t+t` 
	- tensor addition
- `s+s` 
	- scalar addition
- `t-t`
	- tensor subtraction
- `s-s`
	- scalar subtraction
- `-t`
	- tensor negation
- `-s`
	- scalar negation
- `t*t`
	- tensor-tensor multiplication
- `t*s`
	- tensor-scalar multiplication
- `s*t`
	- scalar-tensor multiplication
- `s*s`
	- scalar-scalar multiplication
- `t=t`
	- tensor copying
- `s=s`
	- scalar copying
- `s=const`
	- context operation, extra parameter: `float`
	- set a scalar to a constant
- `sin`
	- sine function
- `cos`
	- cosine function
- `tanh`
	- tanh function
- `tanh_gradient`
	- tanh gradient function
- `sigmoid`
	- sigmoid function
- `sigmoid_gradient`
	- sigmoid gradient function
- `tranpose`
	- tensor transpose
- `element_mult`
	- element-wise tensor multiplication
- `square_loss`
	- square loss function
- `softmax`
	- softmax function
- `clip`
	- clip function to the range `[-1, +1]`
- `label_entropy_loss`
	- entropy loss function given the ground-truth label
- `label_softmax_entropy_gradient`
	- entropy gradient function given the ground-truth label
- `zero_clear`
	- clears all entries in the tensor to be 0
- `fill_rand`
	- fills the tensor with random numbers 
	- for neural network parameter initialization
	- typically a uniform distribution; occasionally Gaussian distribution with a zero mean and a very small standard deviation
- `fill_element`
	- context operation, extra parameter:
	 `lmn::ElementFillFunction<float>`
	- fills the tensor entries by a "filler" function
	- The filler takes a `DimIndex` as argument and returns a `float`
- `set_value`
	- context operation, extra parameters: `DimIndex`, `float`
	- sets the value at `DimIndex` to be a specified `float`
- `perturb`
	- context operation, extra parameters: `DimIndex`, `float`
	- perturbs the entry at `DimIndex` by adding `float` to it

All context operations and their required extra arguments are specified above.

Note that you don't have to implement *all* of the above opcodes. 

For example, if you never use `ClippedMomentum`, you don't have to implement `"clip"` opcode; if you don't plan to run gradient check, you don't have to implement `perturb` or `set_value`. 

###Register your functions with the opcodes

For normal operations, you will register in your engine constructor by:

    register_normal_op("t+t", MyImpl::add);
    
	register_normal_op("sigmoid", MyImpl::sigmoid);
	
	register_normal_op("-s", MyImpl::negate_scalar);

For context operations, you will register in your engine constructor by:

	register_context_op<DimIndex, float>(
		"perturb", MyImpl::perturb);
	
	register_context_op<float>(
		"s=const", MyImpl::set_const);

Specify the extra context argument types in the variadic template of `register_context_op`. 
 
###Add your own opcodes

The built-in *Laminar* opcodes are not special in any sense. You can add your own normal/context opcodes as long as you properly implement and register them in your engine. 

#Learning Modules

##LearningSession

The central class in the learning modules is `LearningSession`. It is a generic class parameterized by 6 modules. 

    template<typename OptimizerT,
		typename EvaluatorT,
		typename StopCriteriaT = MaxEpochStopper,
		typename SerializerT = NullSerializer,
		typename EvalScheduleT = NullSchedule,
		typename ObserverT = NullObserver>
	class LearningSession;

Instead of calling the constructor of this generic class, you should use the generic function `new_learning_session()` because it allows automatic template deduction and default arguments. 

The module order follows the ***O-E-S-S-E-O*** mnemonic. 

##LearningState

A `LearningState` object passes important information during the learning process to various modules. 

It currently has the following fields:

- `int epoch`
	- current epoch number
- `int batchInEpoch`
	- current minibatch number in this epoch
- `int batchAll`
	- all minibatches processed so far
- `int batchSize`
	- current minibatch size
- `float curBatchTrainingLoss`
	- average training loss per example in the current minibatch
- `float validationLoss`
	- validation loss
- `float validationMetric`
	- validation performance metric (e.g. percentage classification error rate)
- `float testingLoss`
	- testing loss
- `float testingMetric`
	- testing performance metric (e.g. percentage classification error rate)

##Optimizer

There are 6 built-in optimizers shipped with the library.

All derived classes from `Optimizer` base class implements the following pure virtual methods:

    // optimizer initialization
    virtual void initialize_impl() = 0;

	// set up the history with respect to a parameter
	// if your optimizer is a history-based algorithm
	virtual void setup_impl(Tensor& paramValue) = 0;

	virtual void update_impl(
		Tensor& paramValue, 
		Tensor& paramGradient, 
		LearningState::Ptr state) = 0;

###SGD

Stochastic gradient descent.

###MomentumGD

Momentum gradient descent

###ClippedMomentum

Momentum descent with gradients clipped within `[-1, 1]`. Typically used for recurrent neural networks to alleviate their numerical unstability. 

###NesterovMomentum

An advanced version of momentum gradient descent. 

###Adagrad

Adaptive gradient that normalizes the descent step by its history. 

###RMSprop

"Root-mean-square Propagation". A more advanced version of Adagrad invented by Geoffery Hinton. 

##Evaluator

All derived classes from `Evaluator` base class implements the following pure virtual methods:

    // Clean up and start new evaluation
	// will be called at the very beginning of an evalution run
	virtual void start_metric(
		Network::Ptr, 
		std::shared_ptr<EngineT>, 
		LearningPhase) = 0;

	// Derived needs to implement this
	// will be called after every minibatch to update whatever metric
	virtual void update_metric(
		Network::Ptr, 
		std::shared_ptr<EngineT>, 
		LearningPhase) = 0;

	// will be called at the end of validation/testing to get a summary metric
	virtual FloatT summarize_metric(
		Network::Ptr, 
		std::shared_ptr<EngineT>, 
		LearningPhase) = 0;

##StopCriteria

All derived classes from `StopCriteria` base class implements the following pure virtual methods:

    // return true to stop learning
    virtual bool stop_learning(LearningState::Ptr) = 0;

Currently *Laminar* ships with a simple `MaxEpochStopper`:

###MaxEpochStopper

Stops the learning after a specified maximal number of epochs reached. 

##Serializer

All derived classes from `Serializer` base class implements the following pure virtual methods:

    virtual void save_impl(
		std::shared_ptr<NetworkT>,
		std::shared_ptr<EngineT>, 
		LearningState::Ptr) = 0;

###NullSerializer

Default placeholder that does nothing.

##EvalSchedule

All derived classes from `EvalSchedule` base class implements the following pure virtual methods:

    // Called at the end of every epoch
	// return true if you want to run validation
	
	virtual bool run_validation(LearningState::Ptr) = 0;

	// Called at the end of every epoch (epoch not incremented yet)
	// return true if you want to run testing
	
	virtual bool run_testing(LearningState::Ptr) = 0;

###NullSchedule

Does not do either validation or testing. 

`NullSchedule` is the default if you omit the argument. 

###NoValidationSchedule

Does not run validation, only testing. 

###NoTestingSchedule

Does not run testing, only validation

###EpochIntervalSchedule

Specify how often do you want to run validation and testing. For example, validation after every 3 epochs and testing after every 10 epochs. 

##Observer

All derived classes from `Observer` base class implements the following pure virtual methods:

    virtual void observe_impl(
	    std::shared_ptr<NetworkT>, 
	    LearningState::Ptr) = 0;

###NullObserver

Does not do anything. `NullObserver` is the default if you omit the argument. 

###MinibatchObserver

Prints out loss information after every minibatch. 

#Exception

*Laminar* has a hierarchy of runtime exceptions. 

![exceptions](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/exceptions.png)

##Laminar Exception

`LaminarException` directly extends `std::exception` and is the highest level exception type in the hierarchy. All runtime exceptions thrown are derived from `LaminarException`.

It is rarely thrown by itself, except when the error is too hard to be classify into a more specific category. 

##Network Exception
A `NetworkException` is thrown by the `Network` base class, or any of its derived classes, e.g. `ForwardNetwork` and `RecurrentNetwork`.

The exception is typically thrown when there are inconsistencies in the network topology. 

Situations that trigger `NetworkException` include, but are not limited to, the following situations:

- A non-recurrent connection attempts to connect a layer to itself. <br><br>
- The network has no input layer.<br><br>
- The network has no loss layer (the last layer must be a derived class of `LossLayer`)<br><br>
- For recurrent networks, an error occurs when a connection's temporal skip value exceeds the global `maxTemporalSkip`. <br><br>

##Component Exception

A `ComponentException` is thrown when either `Layer` or `Connection` detects an inconsistency. 

Situations that trigger `ComponentException` include, but are not limited to, the following situations:

- `Layer` receives an invalid `Dimension` <br><br>
- `ConstantConnection` attempts to connect two layers with different dimensions. <br><br>
- `GatedConnection` dimension mismatch.

##Engine Exception

An `EngineException` is thrown whenever an internal inconsistency occurs in either the virtual engine itself or any of its backends. 

##Tensor Exception

A `TensorException` is thrown when invalid operations are performed on a tensor.

##Learning Exception

A `LearningException` is thrown when the learning procedure detects a logical inconsistency.

Any of the ***O-E-S-S-E-O*** modules (Optimizer, Evaluator, Stop criteria, Serializer, Evaluation schedule, Observer) can throw a `LearningException`.

##Data Exception

A `DataException` is normally thrown by the `DataManager`. 

Any invalid IO operations that read/load training/testing data will trigger a `DataException`. 

When the training/testing data streams are depleted but the user fails to reset a new epoch, `DataException` will also be triggered. 
