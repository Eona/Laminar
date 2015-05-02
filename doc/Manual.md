#Laminar Manual

##Network Topology

###Network
*Laminar* framework currently supports two major networks, feed forward net `ForwardNet` and `RecurrentNet`, both extends the `Network` super class. 

`ForwardNet` has the following methods to build an arbitrary topology:

####Layer

`add_layer(Layer::Ptr)` 

Adds an existing layer pointer to the network architecture. 

####Connection
`add_connection(Connection::Ptr)` 

Adds an existing connection to the network topology. 

`new_connection<ConnectionT>(Layer::Ptr in, Layer::Ptr out)` 

Constructs a new connection internally, and is a convenience method that's equivalent to 

    auto conn = Connection::make<ConnectionT>(in, out);
    net->add_connection(conn);

####Bias
`new_bias_layer(Layer::Ptr)` 

Constructs a new bias layer and adds it to the layer being passed in as a parameter.

The bias layer will have exactly the same dimension as the input layer. 

It is equivalent to 

    auto bias = Layer::make<BiasLayer>(inLayer->dim());
    net->add_connection(bias, inLayer);
    net->add_layer(bias);

####Composite
There are two flavors of `add_composite()`:

    template<typename CompositeT>
	void add_composite(std::shared_ptr<CompositeT> composite);

and 

    template<typename CompositeT>
	void add_composite(CompositeT& composite);

You can either create a `shared_ptr` of `Composite` or an object of `Composite` and pass in a reference. 

The method adds a `Composite` that automates subgraph generation for the network.

The most notable use case of `Composite` is `LstmComposite`. LSTM networks have extremely complicated topology involving almost a dozen layers, `FullConnection` and multiple `GatedConnection` and `GatedTahnConnection`. 

While you can always construct an LSTM from scratch, the `Composite` class greatly improves your productivity. 

###`RecurrentNetwork`

A `RecurrentNetwork` has the following new methods:

####Temporal skip

`init_max_temporal_skip(int)` 

Sets the maximal temporal skip value.

If you set the temporal skip value to a special `Layer::UNLIMITED_TEMPORAL_SKIP` constant, the network will save the full gradient history so that you can jump to an arbitrary frame back in time. 

It is recommended that you keep the value as low as possible, because the more temporal skip you set, the more gradient history the network will save. This might result in unnecessarily large memory footprint. `Layer::UNLIMITED_TEMPORAL_SKIP` is typically used for learning debugging, if you wish to inspect the gradient history at every time frame. 

####Recurrent connection
`add_recur_connection(Connection::Ptr)` 

Adds an existing recurrent connection that has default temporal skip value = 1. 

`new_recur_connection<ConnectionT>(Layer::Ptr in, Layer::Ptr out)`

Constructs a new recurrent connection that connections the *t - 1* frame of `inLayer` to the current frame *t* of `outLayer`. 

The method is equivalent to 

    auto conn = Connection::make<ConnectionT>(inLayer, outLayer);
    net->add_recur_connection(conn);

####Temporal skip connection
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

##Layers

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

`void forward_impl(Tensor& inValue, Tensor& outValue)`

`void backward_impl(
Tensor& outValue, Tensor& outGradient, Tensor& inValue, Tensor& inGradient)`

The `forward_impl` and `backward_impl` are the actual forward/backward propagation logic. All subclasses of `Layer` implement this interface.

Each `Layer` (except for the special `BiasLayer`) takes at least one constructor argument: `Dimension dim`, with `Dimension` as a type alias for `vector<int>` to describe the dimensionality of tensors. 

###ConstantLayer

`ConstantLayer` is a data-holding layer that does not do any computation. Its `forward_impl` and `backward_impl` do basically nothing. 

What is more, it overrides the getter methods:

`Tensor& out_value(int frame)`
`Tensor& out_gradient(int frame)`
`Tensor::Ptr out_value_ptr(int frame)`
`Tensor::Ptr out_gradient_ptr(int frame)`

such that `out_value` redirects to `in_value` and `out_gradient` redirects to `in_gradient`. This approach saves unnecessary copying and memory footprint.

###Activation Layer
####SigmoidLayer
 Computes the sigmoid function
`1 / (1 + exp(-x))`

This is the most commonly used non-linear activation function.

####TanhLayer
Computes the tanh function

`tanh(x)  = (exp(2*x) - 1) / (exp(2*x) + 1)`

This is the second most commonly used non-linear activation function and the default function used in `GatedConnection`. It's commonly the nonlinear gate choice for LSTM networks.

####ScalarLayer
Computes a simple linear scaling function

`f(x) = x * scalar`

####DebugLayer

For debugging only.

###Loss Layer
Two plug-in loss layers are shipped with *Laminar* framework. 

####SquareLossLayer

Computes the loss function

`loss(x, y) = (x - y) ** 2`

`SquareLossLayer` is typically used for regression tasks. 

####LabelSoftmaxEntropyLayer

Computes the loss function

`loss(x, y) = 
-log(softmax(x) * I[y == ground_truth_label])`

`LabelSoftmaxEntropyLayer` is typically used for classification task with one-hot encoded labels.

#Virtual Engine Backend
##Tensor
`Tensor` is just a fancy name for multi-dimensional arrays. As for most forward and recurrent networks, `Tensor` is synonymous with matrix. 

`Tensor` does not actually do computation. Its "big six" operators, as well as most common arithmetic operators are all overloaded. Each overload uploads an `Instruction` to the virtual engine instruction queue. If the instruction opcode is `create`, the virtual engine will allocate a new memory on its internal memory pool. 

##Instruction
An instruction has four parts:

###Opcode
A class that wraps a string, the opcode identifies which operation to perform next. 

A detailed listing of all operations required of backend engines will be listed and explained in the next session. 

###Read addresses

Read address is a field in `Instruction`, with the type `std::vector<int>`. Note that each memory address is an integer that refers to a location in the internal memory pool. 

Reading can have multiple addresses. For example, adding two numbers require two operands, so the read address will have length 2.

###Write address

In the current implementation, an operation can write to one and only one write address. 

###OpContext

Some operations cannot perform with only information stored in Tensors. Thus you have to pass in an extra `OpContext` (i.e. operational context) to the `Instruction`

The `OpContext` class uses very special recursive template metaprogramming method to achieve data type heterogenuity. 

In other words, you can pass in *any* number of *any* data type as an `OpContext`, which will be internally stored as an `std::tuple`.


##`Engine<DataType>`

`Engine` is the intermediate agent between the virtual engine and the actual backend implementation. 

###Using built-in backends

There are 6 built-in backends shipped with *Laminar* library. Using them is as easy as changing a single line of code:

   `EngineBase::make<EigenEngine>();`
   
   `EngineBase::make<VecmatEngine>();`
   
   `EngineBase::make<OpenclEngine>(true); // OpenCL on GPU`
   
   `EngineBase::make<OpenclEngine>(false); // OpenCL on CPU`
   
   `EngineBase::make<CudaEngine>();`
   
   `EngineBase::make<CublasEngine>(); // can take an optional memory profiler arg`

###Rolling Your Own Backend

Your own backend engine will work seamlessly with *Laminar* `Network` frontend as long as it conforms to a specific interface. 

All non-`OpContext` operations should have the following signature:

    void op(vector<DataType> reads, DataType write, bool is_initialized);

The list of such operations are:

`assign_t`

Copies a tensor to another tensor

`assign_s`

Copies a scalar to another scalar

`add_t_t`

Tensor addition

`add_s_s`

Scalar addition

`add_t_s`

Element wise scalar addition

`sub_t_t`

Tensor subtraction

`sub_s_s`

Scalar subtraction

`element_mult`

Element-wise multiplication

`element_divide`

Element-wise division

`mult_t_t`

Tensor multiplication

`mult_t_s`, `mult_s_t`

Tensor-scalar multiplication

`mult_s_s`

Scalar-scalar multiplication

`transpose`

Matrix transpose

`sigmoid`

Sigmoid function

`tanh`

Tanh function

`sin`

Sine function

`cos`

Cosine function

`sigmoid_gradient`

Sigmoid's gradient function

`tanh_gradient`

Tanh's gradient function

`clip`

[-1, 1] clipping function

`softmax`

Softmax classification function

`label_entropy_loss`

Cross-entropy loss based on ground-truth labels. 

`label_softmax_gradient`

Softmax gradient function based on ground-truth label

`perturb`

Perturb a single element. Primarily used for gradient checking. 

`zero_clear`

Clears all entries of a Tensor to 0

`fill_rand`

Fills all entries of a Tensor to random numbers

`fill_element`

Fills all entries of a Tensor by a specific filler function. 
