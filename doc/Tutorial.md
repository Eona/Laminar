#Laminar Tutorial

##Introduction
Deep learning is a branch of machine learning based on a set of algorithms that attempt to model high-level abstractions and patterns in data by using non-linear architectures known as _neural networks_. 

Deep learning has seen tremendous success recently in a wide range of domains, from semantic modeling in natural language processing to object recognition and classification in computer vision. Deep neural networks are able to solve extremely hard AI problems not tractable by traditional statistically learning algorithms like support vector machines. Their success can be attributed to a single most powerful theorem in machine learning: the _Universal Approximation Theorem_, which states that 

- Any computable vector function can be approximated by a feed-forward neural network to arbitrary precision.<br><br>
- Any sequential Turing machine can be simulated by a recurrent neural network (RNN) to arbitrary precision.

This theorem looks very promising, but as is usually the case in math, the beauty is limited only in theory. In practice, the implementation details and training procedure are crucial to the performance of a neural network.  

Our library aims to provide a comprehensive framework to train and deploy feed-forward neural networks and recurrent neural networks, two of the most important deep learning architectures. 

The library is named **Laminar**. In fluid dynamics, the term _Laminar flow_ means a steady flow without turbulence. Deep learning is based on _gradient flow_ that forward/backward propagates through the neural network, so we appropriately steal the concept from physics. 

The name is also a recursive acronym:
__*Laminar Accelerated and MInimalistic Neural ARchitecture*__ 

*Laminar* library is designed to be:

- User-friendly: as you will see in the next few chapters.<br><br>
- Efficient: the library runs very fast on both CPU and GPU, using state-of-the-art computational backends. <br><br>
- Versatile: provides a large number of built-in pluggable modules. <br><br>
	- Most common networks can be constructed and trained with ease.<br><br>
	- Six computational backends have been provided, which fully support most mainstream CPUs and GPUs (even ones without CUDA support). <br><br>
- Customizable in every corner: if the built-in modules cannot satisfy your needs, you can always roll your own modules by extending the architecture. <br><br>

This tutorial assumes that readers are reasonably familiar with the basic concepts of machine learning, e.g. loss function, gradient descent, training, cross-validation, etc. [This website](http://deeplearning.net/) is a good place to start with these fundamental concepts. 

## Installation

_Laminar_ is a C++ 11 template-based header library, which means there is no installation process - simply unzip the source and include in your project. This eliminates the hassle of linking dynamic libraries.  

_Laminar_ relies heavily on C++ 11 features like variadic template, `auto`, `decltype`, and lambdas. We have only tested it on Ubuntu 14.04 with g++ 4.8. We will test the framework on Mac and Windows soon. 

If you use the default computation backend, the only dependency is the C++ standard library. If you want to enable GPU mode, you must install CUDA toolkit from NVidia, which includes CUDA, cuBLAS and OpenCL. 

The minimal CUDA version required is 6.5, because earlier CUDA does not support C++ 11. CUDA 7.0 has the best C++ 11 compatibility, so it is strongly recommended that you install the latest CUDA toolkit available (7.0 at the time of this writing). 

_Laminar_ uses `cmake` to manage the external library dependencies. The minimal `cmake` version required is 2.8. 
`cmake` will try to find the libraries in their default places. If not found, you will have to specify the library paths by passing command line options. For example, to specify CUDA directory:

    cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda

Happy deep learning!

#Architecture
<img src="https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/overview.png" alt="Overview" height="400" width="200">

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


#Feed-forward Neural Network
##Create your first network

![Components](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/components.png)

The network is essentially a directed acyclic graph (DAG). The _vertex_ in a graph is a _Layer_ in _Laminar_'s terminology, and _edge_ is a _Connection_.

Both _Layer_ and _Connection_ are subclasses of _Component_. The above graph is an incomplete overview of the component hierarchy. For example, there are currently five subclasses that extend `ActivationLayer`, e.g. `SigmoidLayer`, `CosineLayer`, etc. 

For more technical details about specific layers and connections, please refer to the _Laminar_ manual. 

Let's build the simplest feed-forward neural network with one hidden layer. 

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/forward_s.png)

The graph illustrates the basic requirements of constructing a neural network:

 - Input layer: known in the code as `ConstantLayer`. This layer is the entry point to all networks. <br><br>
	 - It contains the data loaded by `DataManager`, a concept that will be covered later. <br><br>
	 - We choose not to call the layer `InputLayer` because `ConstantLayer` can also be used as a hidden layer that does nothing but passing data unaltered. It is useful in constructing *Long Short Term Memory* recurrent networks, typically refered to as *LSTM RNN* in deep learning literature. <br><br>
 - Hidden layer: can be arranged in any configuration.<br><br>
 - Loss layer: all networks must have one and only one loss layer. <br><br>
  - Loss layer computes the loss function, which is the optimization objective of the neural network training procedure. <br><br>
  - Typical loss layers are <br><br>
		 - `SquareLoss` is typically used in regression tasks. It computes the square difference between neural network's output and the objective function. <br><br>
		 - `CrossEntropyLoss` is more often used in discrete class classification tasks. It computes the cross entropy function at the correct class label. <br><br>
 - Connection: the most commonly used connection in deep learning is `FullConnection`, which contains learnable parameters between two fully-connected layers. <br><br>
	Later we will see more complicated connections like `GatedConnection`, a fundamental building block for RNNs. 

Let's look at the code that constructs the above graph. 

First, we need to include the necessary headers. Assume you place the library in a directory `laminar` on system path:

    #include <laminar/network.h>
    #include <laminar/full_connection.h>
    #include <laminar/activation_layer.h>

Then we can proceed to construct the neural network. 

	// construct a forward network
	// parameters "engine" and "dataManager" will be covered later
	auto net = ForwardNetwork::make(engine, dataManager);
	
	// create layers
	auto inputLayer = Layer::make<ConstantLayer>(20);
	auto hiddenLayer = Layer::make<SigmoidLayer>(100);
	auto lossLayer = Layer::make<SquareLossLayer>(3);
	
	// add layers and connections
	net->add_layer(inputLayer);
	net->add_connection(Connection::make<FullConnection>(inputLayer, hiddenLayer));
	net->add_layer(hiddenLayer);
	net->add_connection(Connection::make<FullConnection>(hiddenLayer, lossLayer));
	net->add_layer(lossLayer);


In _Laminar_, all components are constructed by using the static method `::make<ComponentType>(constructor_args ...)`. 

It is implemented by C++ 11 variadic template. The syntax is similar to `std::make_shared<>()`. In fact, the method returns a shared pointer to the component. For example, line

    Layer::make<SigmoidLayer>(100);  

returns a smart pointer `std::shared_ptr<SigmoidLayer>`, which has a `SigmoidLayer` of dimension 100. 

Some layers like `ScalarLayer` might take more than one constructor argument:

    // 130-dimensional linear scaling layer, multiplier = 1.34f
    Layer::make<ScalarLayer>(130, 1.34f);

Because shared pointers are so commonly used, _Laminar_ has special typedefs to make the code more readable. 

    // the following two types are equivalent
    std::shared_ptr<Layer>
    Layer::Ptr
  
Thanks to C++ 11 `auto` keyword, you don't have to write either of the above explicitly. But if you want to extend the library architecture, you will see `XXX::Ptr` very frequently. 

A `Connection` always has an in-layer and an out-layer. It is specified in the `Connection` constructor.

    Connection::make<FullConnection>(inLayer, outLayer);

The above line returns a `Connection::Ptr`.

To add the components to the neural network, you call the corresponding methods in topological order (i.e. order of the DAG arrow). 

    net->add_layer(Layer::Ptr)
    net->add_connection(Connection::Ptr)

Note that the last statement should always be 

    net->add_layer(lossLayer)

Because `Network` assumes that the terminal layer is a loss layer that computes the network objective function. 

For your convenience, we have introduced a shortcut

    net->new_connection<ConnectionType>(inLayer, outLayer);
    
    // exactly equivalent to
    auto conn = Connection::make<ConnectionType>(inLayer, outLayer);
    net->add_connection(conn);

to add anonymous connections. We have made extra efforts to keep naming conventions consistent throughout the massive codebase of _Laminar_. Here is one example. Any method prefixed with `add_` adds an existing object/pointer, while a method prefixed with `new_` constructs an anonymous object/pointer internally. 
For most cases, `new_XXX()` is nothing but a convenient shortcut for `add_XXX()`

Note that `net->new_connection<ConnectionType>()` is yet another example of how C++ 11 variadic template can be employed to simplify _Laminar_'s syntax. 

In addition to make, there is also a static class method `::cast<DerivedType>` for abstract classes like `Layer` and `Connection`:

    auto derivedPtr = Layer::cast<SigmoidLayer>(layerPtr);

The cast returns a null shared pointer if the types are incompatible. 

##Gearing up feed-forward network

With the basic concepts in mind, we are ready to create more complicated network structures. 

![s_s_bias](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/forward_s_s.png)

The above network has a new layer type, `BiasLayer`, which adds bias weights to feed forward connections. 

`BiasLayer` is somewhat special because it *always* has only outgoing connection. It is an additive layer that has the same activation all the time, specifically a row matrix `[1, 1, 1 ...]` with all ones. 

Because `BiasLayer` is so special, it has its own convenience method `new_bias_layer()` as a member function in `Network`. 

Let's construct this network:

	// extra include
    #include <laminar/bias_layer.h>

	// construct a forward network
	// parameters "engine" and "dataManager" will be covered later
	auto net = ForwardNetwork::make(engine, dataManager);
	
    // create layers
	auto inputLayer = Layer::make<ConstantLayer>(20);
	auto hidden1 = Layer::make<SigmoidLayer>(100);
	auto hidden1 = Layer::make<SigmoidLayer>(100);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(3);
	
	// add layers and connections
	net->add_layer(inputLayer);
	net->new_connection<FullConnection>(inputLayer, hidden1);
	// hidden layer 1's bias
	net->new_bias_layer(hidden1);
	net->add_layer(hidden1);
	net->new_connection<FullConnection>(hidden1, hidden2);
	// hidden layer 2's bias
	net->new_bias_layer(hidden2);
	net->add_layer(hidden2);
	net->new_connection<FullConnection>(hidden2, lossLayer);
	net->add_layer(lossLayer);

The convenience method 

    net->new_bias_layer(hidden1);

is exactly equivalent to

    auto biasLayerHidden1 = Layer::make<BiasLayer>(hidden1->dim());
    net->add_bias_layer(biasLayerHidden1);
    net->new_connection<FullConnection>(biasLayerHidden1, hidden1);

Note how much typing we have saved. 

The `BiasLayer` always has the same dimension as the layer being biased. In our current release, only fully connected bias layers (`FullConnection`) are supported. In future releases, we will have a templated `add_bias_layer()` method that supports more connection types. 

`LabelSoftmaxEntropyLayer` is the cross entropy loss layer used for labeled discrete classification. 

As mentioned in the introduction chapter, *Laminar* supports arbitrarily complicated network topology as long as you can describe it using `add_layer()` and `add_connection()`. The types of networks you can create are only limited by your imagination. 

For example, *Laminar* can build networks not yet studied in the deep learning literature, as far as we know. The following "diamond" forward network is an example. 

![diamond_forward](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/diamond_forward.png)

Traditional forward networks are mostly one-dimensional, e.g. layer one to two, two to three, three to four, etc. The diamond network above is a novel two-dimensional graph topologies. Its learning properties are not studied in depth. *Laminar* library can help you create new topologies and make new discoveries. 

The code for the above network should look straightforward by now. 

    // create layers
	auto inputLayer = Layer::make<ConstantLayer>(20);
	auto tanhHidden1 = Layer::make<TanhLayer>(120);
	auto tanhHidden2 = Layer::make<SigmoidLayer>(96);
	auto sigmoidHidden = Layer::make<SigmoidLayer>(150);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(3);
	
	// add layers and connections
	net->add_layer(inputLayer);
	net->new_connection<FullConnection>(inputLayer, tanhHidden1);
	net->new_connection<FullConnection>(inputLayer, tanhHidden2);
	// diamond sides
	net->add_layer(tanhHidden1);
	net->add_layer(tanhHidden2);
	net->new_connection<FullConnection>(tanhHidden1, sigmoidHidden);
	net->new_connection<FullConnection>(tanhHidden2, sigmoidHidden);
	// diamond top
	net->new_bias_layer(sigmoidHidden);
	net->add_layer(sigmoidHidden);
	net->new_connection<FullConnection>(sigmoidHidden, lossLayer);
	net->add_layer(lossLayer);

Each line above specifies either a layer (node) or a connection (edge). No code is wasted - in other words, these are the minimal codes required to build your dream network. 

#Recurrent Neural Network (RNN)
##TitleCreate your first RNN

While the component building blocks are the same as feed-forward networks, RNNs have an extra temporal dimension, which makes them dramatically more powerful (equivalent to generic Turing machine) and at the same time much harder to train.

RNN graphs are no longer DAGs because they can have self-loops and mutually-recurrent loops. What is more, they are not even 2D anymore, thanks to the added temporal dimension. 

_Laminar_ is superior to most other open source RNN implementations because it allows arbitrary connection over *time*. Traditional RNNs can only back propagate to its state at *t - 1*, given current time frame *t*. 
_Laminar_ RNNs can propagate to *any* time frame in history (*t - 2*, *t - 3*, ..., all the way back to the time of Adam and Eve). Due to lack of terminology in the current deep learning literature, we call this class of RNNs "*time-skip RNN*", which travel back in time more than one frame. 

To use RNNs, you need to include one more header:

    #include <laminar/rnn.h>

Let's start with the simplest traditional RNN.

![simple RNN](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/simple_rnn.png)

The remarkable difference from forward networks is the self-loop on the two sigmoid hidden layers, numbered with a *temporal skip* value. Temporal skip has been informally defined above as "how many frames to travel back in time". Because the default temporal skip is 1, we don't have to specify anything special. 

    // construct a recurrent network
	// parameters "engine" and "dataManager" will be covered later
	auto net = RecurrentNetwork::make(engine, dataManager);
	
	// create layers
	auto inputLayer = Layer::make<ConstantLayer>(20);
	auto hidden1 = Layer::make<SigmoidLayer>(100);
	auto hidden2 = Layer::make<SigmoidLayer>(100);
	auto lossLayer = Layer::make<SquareLossLayer>(3);
	
	net->add_layer(inputLayer);
	// same as before
	net->new_connection<FullConnection>(inputLayer, hidden1);
	net->new_bias_layer(hidden1);
	// recurrent connection
	net->new_recur_connection<FullConnection>(hidden1, hidden1);
	net->add_layer(hidden1);
	// same as before
	net->new_connection<FullConnection>(hidden1, hidden2);
	net->new_bias_layer(hidden2);
	// recurrent connection
	net->new_recur_connection<FullConnection>(hidden2, hidden2);
	net->add_layer(hidden2);
    net->new_connection<FullConnection>(hidden2, lossLayer);

As usual, the following convenience method

    net->new_recur_connection<ConnectionType>(inLayer, outLayer);

is exactly equivalent to

    auto conn = Connection::make<ConnectionType>(inLayer, outLayer);
    net->add_recur_connection(conn);

Note that a recurrent connection can connect a layer to itself (`inLayer == outLayer`), while a normal connection cannot. 

##Time-skip RNN

We are ready to build an RNN that travels arbitrary along the temporal dimension. 

![Time skip RNN](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/rnn_time_skip.png)

Compare this to the RNN we have just implemented above. Instead of one self-loop, each hidden layer has two self-loops, one with temporal skip = 1 and the other = 2. 

The implementation is very similar to the previous section, except that you have to specify the maximal temporal skip value explicitly after constructing a `RecurrentNetwork` object:

    net.init_max_temporal_skip(3);

If you set the temporal skip value to a special `Layer::UNLIMITED_TEMPORAL_SKIP` constant, the network will save the full gradient history so that you can jump to an arbitrary frame back in time. 

It is recommended that you keep the value as low as possible, because the more temporal skip you set, the more gradient history the network will save. This might result in unnecessarily large memory footprint. `Layer::UNLIMITED_TEMPORAL_SKIP` is typically used for learning debugging, if you wish to inspect the gradient history at every time frame. 

To add a temporal skip = 2 connection, 

    net->new_recur_skip_connection<FullConnection>(2, hidden1, hidden1);
    net->new_recur_skip_connection<FullConnection>(3, hidden2, hidden2);

The first `int` argument is the temporal skip value of this connection. As you may have guessed, the following are exactly equivalent:

    net->new_recur_skip_connection<FullConnection>(1, hidden1, hidden2);
    // same as
    net->new_recur_connection<FullConnection>(hidden1, hidden2);

The layers can also be connected to each other across time.

![mutually recurrent](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/mutually_recurrent_rnn.png)

The two `TanhLayer` are both self-recurrent and mutually-recurrent. 

## Gated RNNs

Gated RNNs are not very interesting in themselves - their full glory is only revealed when combined with LSTM RNNs, the subject of the next section. To prepare for the climactic moment, we start with a relatively simple gated RNN:

![Gated RNN](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/gated.png)

The dashed line means that `CosineLayer` acts as a multiplicative gate that regulates the regular `TanhLayer`. As usual, a `GatedConnection` can have temporal skip values. 

The code to implement the above gated RNN:

    net->init_max_temporal_skip(2);
    net->add_layer(inputLayer);
	// same as before
	net->new_connection<FullConnection>(inputLayer, cosLayer);
	net->new_connection<FullConnection>(inputLayer, tanhLayer);
	// the first gated connection is NOT a temporal connection!
	auto g234 = Connection::make<GatedConnection>(tanhLayer, cosLayer, lossLayer);
	// the following two are temporal connections
	auto g234_1 = Connection::make<GatedConnection>(tanhLayer, cosLayer, lossLayer);
	auto g234_2 = Connection::make<GatedConnection>(tanhLayer, cosLayer, lossLayer);
	net->add_connection(g234);
	net->add_recur_connection(g234_1);
	// specify temporal skip value
	net->add_recur_connection(g234_2, 2);
	net->add_layer(lossLayer);
	
A `GatedConnection` is constructed by three `Layer::Ptr`, in order:
1. Incoming layer
2. Gate layer that regulates the incoming layer
3. Outgoing layer

It can be added as either a regular connection or a recurrent connection. In the above example, both types coexist. 

##Long Short Term Memory RNN (LSTM)

It has been proven that traditional RNNs suffer from a severe learning problem known as *gradient explosion*. This is a numerical stability problem inherent in the RNN training algorithm. 

Researchers have come up with a much more complicated architecture that solves the gradient explosion problem by adding `GatedConnection` that controls the information flow by an adaptive multiplication gate. 

The most successful network based on `GatedConnection` is the LSTM network:

![LSTM](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/lstm.png)

The input gate, forget gate and output gate controls when to store, propagate and clear ("forget") information. It turns out that LSTM networks have much better long-term memory than RNNs, as its name suggests. In practice, they excel in sequential problems like speech recognition, machine translation, DNA sequence analysis, etc. 

You can implement a full-fledged LSTM RNN by adding each layer, full connections and gated connections one by one. This is doable but very tedious, so we provide you with a convenience class `Composite` that automates the LSTM graph construction for you. 

Specifically, you will use the `LstmComposite` class. If you are curious about how LSTM is constructed, you can look into `lstm.h` header file. 

    // extra headers
    #include <laminar/composite.h>
    #include <laminar/lstm.h>
    
	net->add_layer(inLayer);

	auto lstmComposite = 
		Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, 110);

	net->add_composite(lstmComposite);
	net->new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);
	net->add_layer(lossLayer);

When you add a `Composite`, you need to specify its dimension (as usual) and its incoming layer. The composite will add the LSTM graph to your network, and you can then access its last layer by `lstmComposite->out_layer()`. You may connect it directly to a `LossLayer` or to another LSTM / RNN / whatever topology you would like to add on top of it. You can easily create a "diamond LSTM" by adding three `LstmComposite`. As far as we know, *Laminar* is the only library that has such capability. 

## Extending the Topology

Feed forward networks and RNNs have covered a significant part of deep learning. But nothing is enough in the realm of science. If you wish to extend the network topology, please refer to the manual for more information. You can add new `Layer` types, or even `Network` types. 

For example, *Laminar* does not yet support convolutional networks, another important family of deep learning architecture that makes autonomous driving and high-level vision tasks possible. You can extend the architecture by using the Developer API specified in the manual. 

#Training Neural Networks

## Engine

This part is trivial unless you are a developer who wants to roll your own backend engine. This tutorial primarily focuses on the user API. The developer API is covered in the manual. 

As mentioned in the introductory part, the *Laminar* framework ships with 6 compute backends, which should satisfy most of the needs for education and casual researchers. 

The engine can be constructed in one line:

    auto engine = EngineBase::make<EngineType>(constructor_args ...);

 `engine` should be passed to `Network::make` as the first argument. 

## Data Manager

Training and testing data are essential to a learning procedure. We cannot provide a drop-in data manager because training data can be in any format, like xml, plain text, binary data file, images, audio, etc. 

You will have to write custom code to load your training data. Luckily, we have a unified abstract interface to help you stay organized - as long as you extend and implement the pure virtual methods in `DataManager`. 

    #include <laminar/data_manager.h>
	
	struct MyDataMgr : public DataManager<MyDataType>
	{	
		virtual bool prepare_next_batch_impl(LearningPhase) = 0;

		virtual void reset_epoch_impl(LearningPhase) = 0;

		virtual void load_input(DataPtr write, bool is_initialized, LearningPhase) = 0;

		virtual void load_target(DataPtr write, bool is_initialized, LearningPhase) = 0;
	}

 `MyDataType` has to be the same as your engine's data type. A listing of required data types for each backend can be found in the manual. 

`DataPtr` is a type alias for `std::shared_ptr<MyDataType>`

`LearningPhase` is an enum that has three values
- `LearningPhase::Training` for the training phase
- `LearningPhase::Validation` for cross-validation phase
- `LearningPhase::Testing` for testing phase

For each different phase, the data should typically be different. It is actually a common mistake in machine learning to overlap validation/testing data with training data. 

`prepare_next_batch_impl` should prepare the data IO for the next minibatch. It returns a bool, which shoudl be `true` when this batch is the last batch in the specific learning phase. Later you will see that `LearningSession` relies on this boolean to update epochs. 

`reset_epoch_impl` is called when a learning session completes one epoch of learning. You should reset the IO and reload the training/validation/testing data for the next epoch. 

In `load_input()` and `load_target()`, you need to fill the `DataPtr` by the data loaded from your training/validation/testing files. 

## Learning Session

After we set up the engine and data manager, it is finally time for training. 

A `LearningSession` object encapsulates all the logic required to do a complete deep learning experiment. Every part of it is fully customizable - in fact, it is a template class that takes 6 type parameters:

    template<typename OptimizerT,
		typename EvaluatorT,
		typename StopCriteriaT = MaxEpochStopper,
		typename SerializerT = NullSerializer,
		typename EvalScheduleT = NullSchedule,
		typename ObserverT = NullObserver>
	class LearningSession;

Each one represents an important learning component, although some of them can be omitted to use the default implementation. 

Here is a handy palindrome mnemonic to help you organize the modules:

###***O-E-S-S-E-O***
- **O**: &nbsp;&nbsp;&nbsp;&nbsp;Optimizer
- **E**:  &nbsp;&nbsp;&nbsp;&nbsp;Evaluator
- **S**:  &nbsp;&nbsp;&nbsp;&nbsp;Stop criteria
- **S**: &nbsp;&nbsp;&nbsp;&nbsp;Serializer
- **E**: &nbsp;&nbsp;&nbsp;&nbsp;Evaluation schedule
- **O**: &nbsp;&nbsp;&nbsp;&nbsp;Observer

### **O** for Optimizer
An optimizer is a numerical algorithm that tunes the gradient every time after a minibatch back propagation. It is the primary workforce that optimizes the neural network's object function (i.e. loss value), so it deserves to the foremost module. 

6 built-in optimizers are shipped with *Laminar*. You can implement your own optimizer by extending the `Optimizer` abstract base class. 

The most common optimizer is *Stochastic Gradient Descent*, known as `SGD` in *Laminar* terminology. 

    // Learning rate decays every epoch if decayRate < 1.f
    // learning accelerates if decayRate > 1.f
    // default decayRate = 1.f
    auto opm = Optimizer::make<SGD>(learningRate, decayRate); 

It is advised to choose a small learning rate. If it converges too slowly, you can increase the learning rate and repeat the process. 

### **E** for Evaluator

An evaluator runs validation and testing after each training epoch. It encapsulates the logic for calculating validation and testing "metric". 

A metric is not necessarily the same as the network loss function. Take classification task for an example. Suppose you are trying to classify handwritten digits, your *loss value* will be the relative cross-entropy between the predicted distribution and the actual one-hot distribution. But your *metric* is the percentage accuracy of your network's prediction. 

At the end of the day, you will report "98.3% success rate" to your colleague, not "0.05 average cross-entropy loss". 

Because your metric can be anything your task defines, you can implement the `Evaluator` virtual methods to report the number.

### **S** for Stop Criteria

The learning has to stop at some point, for example, after a maximal number of epochs has been reached.

The simplest `StopCriteria` is the built-in `MaxEpochCriteria` that does the above:

    auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);

This module is actually far from trivial. There are quite a few serious papers on when to stop learning in the deep learning community, because early stopping might be a terrible idea. By implement the virtual methods in `StopCriteria`, you can use more advanced stopping techniques described in those papers. 

### **S** for Serializer

If you want to deploy the neural network, you will need to store the learned parameters to disk. This is what a `Serializer` is designed for - in fact, all your effort vanishes into thin air if you lose the parameters tuned after tens of hours or even days of computation time. 

The `Serializer` optionally saves the neural network state after every minibatch. Because we have no idea how you would like to save your parameters, we only provide an abstract interface for you to implement. 

If you only want to run quick experiments, 

    auto ser = NullSerializer::make();

is a placeholder. 

### **E** for Evaluation Schedule

You typically don't want to run validation and testing after every training epoch. Then you can specify an evaluation schedule that tells the learning session when to start validation/testing. 

    auto evalsched = EpochIntervalSchedule::make(validation_frequency, testing_frequency);

### **O** for Observer

If you want to print out any information as the training goes, `Observer` is the plug-in module for you to implement. 

As the author of *Laminar*, I have personally used `Observer` to print out debugging information at every minibatch. 

## Putting together everything

Now that we have learned the ***O-E-S-S-E-O*** workflow, we are ready to put all the pieces together and complete the puzzle. 

The key `LearningSession` object can be constructed by a template function:

    auto session = new_learning_session(net, O_, E_, S_, S_, E_, O_);

Here's a sample *Laminar* experiment that trains a high performance *MNIST* hand-written digit recognizer on cuBLAS:

    auto engine = EngineBase::make<CublasEngine>();
	auto dataman =
		DataManagerBase::make<CublasMnistDataManager>(engine, BATCH_SIZE, "../data/mnist");

	auto linput = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lhidden1 = Layer::make<SigmoidLayer>(300);
	auto lhidden2 = Layer::make<SigmoidLayer>(300);
	auto lloss = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto net = ForwardNetwork::make(engine, dataman);
	net->add_layer(linput);
	net->new_connection<FullConnection>(linput, lhidden1);
	net->new_bias_layer(lhidden1);
	net->add_layer(lhidden1);
	net->new_connection<FullConnection>(lhidden1, lhidden2);
	net->new_bias_layer(lhidden2);
	net->add_layer(lhidden2);
	net->new_connection<FullConnection>(lhidden2, lloss);
	net->add_layer(lloss);

	auto opm = Optimizer::make<NesterovMomentum>(lr, moment);
	auto eval = MnistAccuracyEvaluator::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(0, 1);
	auto obv = NullObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

	session->initialize();
	session->train();

You have just completed your first journey through the *Laminar* framework. 

Congratulations! You are not far from bending deep learning to your will. 


   

