<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
#*Laminar* Performance Report

- [Comparison of the backends](#comparison-of-the-backends)
- [Operation profiling](#operation-profiling)
- [MNIST training](#mnist-training)
  - [Goal](#goal)
  - [Network topology](#network-topology)
  - [Learning](#learning)
- [RNN Language Modeling](#rnn-language-modeling)
  - [Setup](#setup)
  - [Goal](#goal-1)
  - [Corpus](#corpus)
    - [Shakespeare's *A Midsummer Night's Dream*](#shakespeares-a-midsummer-nights-dream)
    - [Kant's *Critique of Pure Reason*](#kants-critique-of-pure-reason)
  - [Network topology](#network-topology-1)
  - [Learning](#learning-1)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

#Comparison of the backends

We have implemented six computation backends and have conducted extensive profiling on their time efficiency. The backends are listed below. 

CPUs:
- Eigen
- Vecmat
- OpenCL (CPU mode)

GPUs:
- CUDA
- cuBLAS
- OpenCL (GPU mode)

We rank them in order of decreasing speed:

1. cuBLAS
2. CUDA
3. OpenCL (GPU mode)
4. OpenCL (CPU mode)
5. Eigen
6. Vecmat

# Operation profiling

In this section, we evaluate the time efficiency performance of individual mathematical operations on matrix in different computation engines. Here is a list of operations we evaluate in thie section:

1. Element-wise addition/subtraction
2. Element-wise multiplication
3. Element-wise Scaling
4. Element-wise Sigmoid 
5. Element-wise Sin
6. Element-wise Cos
7. Matrix Multiplication (no transpose)
8. Matrix Multiplication (left operand transpose)
9. Matrix Multiplication (right operand transpose)

Operation 1-6 are element-wise operations, so they should have similar timing profile. We will focus on analysing operations 7-9 because matrix multiplications are O(n^3) operation with naive implementation.

In this section, the performance of an operation are evaluated based on data throughput, which is defined as the input data size per unit time. All input data used in the experiment are in single-precision floating-point format.

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/throughput.png)

According to the figure, all element-wise operations are roughly 200 times faster than matrix multiplication operations, which are as expected. 

One noticable result is that OpenCL has better performance than Plain CUDA in most of the element-wise operations. 

We believe this is because of the fact that OpenCL is a lower-level interface whereas CUDA is a much abstracted interface, which can cause some overhead. 

The following figure plots the multiplication operations against their data throughput per microsecond. The figure is drawn on log-scale to be more readable. 

![Logscale](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/performance_mult.png)

To exaggerate the performance, a linear-scale figure is also provided:

![Linearscale](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/linear_performance.png)

In terms of matrix multiplication, there is also a gap between the GPU-based engine and CPU-based engine in terms of data throughput. For example, Plain CUDA and is about 100 times faster than Eigen and VecMat. 

One noticeable result is that in CPU-based engines (except Eigen) and OpenCL(GPU) engine, Matrix Multiplication (left operand transpose) is faster than Matrix Multiplication (no transpose), which is also faster than Matrix Multiplication (right side transpose). This is due to the fact that without optimization, the algorithm loops through the matrix and multiply the elements in whichever way the matrix is ordered. 

For example, in no-transpose matrix multiplication operation, the stride size for the right operand is 1, but the stride size of the left operand is the number of rows, which makes memory access extremely inefficent. Matrix Multiplication (right side transpose) is the worst because the stride width for both operand are the number of rows.

Plain CUDA has non of these overhead because the kernel loads the matrix tile to shared memory first using individual kernels, and then it computes the data on the shared memory, which incurs no overhead whatsoever. Eigen does not have this overhead either because of its optimization

Finally, cuBLAS's matrix multiplication is clearly suprior than any other implementations because of it's a heavily optimized linear algebra library.

#MNIST training

##Goal

MNIST is a popular database of handwritten digits with ground-truth labels. In fact, it is widely considered as a touchstone for machine learning algorithm benchmarking.  

Our goals is to classify the images as accurately and fast as possible.

##Network topology

The following are the network topologies we tried in our experiments:

<img src="https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/forward_s_t_s.png" width="200"/>

![forward_s_s](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/forward_s_s.png)

![diamond](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/diamond_forward.png)

##Learning
In terms of testing accuracy, the diamond network reaches a percentage classification accuracy as high as 98.3% after 12 epochs. 

The learning curves of the above three models are shown in the figure below:

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/mnist_learning.png)

A zoomed-in version of the figure shows the learning differences more clearly:

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/mnist_learning_zoomed.png)

Classification percentage accuracy graph:

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/mnist_accuracy.png)

Using cuBLAS backend, the diamond network converges to 98.3% testing accuracy in less than 75 seconds on my humble GT-740 Nvidia GPU. 

This is arguably a state-of-the-art result. 

#RNN Language Modeling

##Setup

We limit ourselves to 64 characters instead of the full ASCII alphabet. 

The 64 charaters include:

- Lower-case letters
- Upper-case letters
- Punctuations
- Commonly used special characters in our training corpus, like parenthesis

We use one-hot encoding and cross entropy loss. 

The training corpus are pre-processed by python scripts. 

##Goal

Our goal is to predict the next character based on the history sequence of characters. 

The perplexity of the predictive distribution should be as low as possible. 

##Corpus

###Shakespeare's *A Midsummer Night's Dream*

Excerpt:

> Stand forth, Lysander: and my gracious duke, <br>
	This man hath bewitch'd the bosom of my child; <br>
	Thou, thou, Lysander,  <br>
	thou hast given her rhymes, <br> 
	And interchanged love-tokens with my child:  <br>
	Thou hast by moonlight at her window sung


###Kant's *Critique of Pure Reason*

Excerpt:

> It is plainly not the effect of the levity, 
but of the matured judgement of the age, 
which refuses to be any longer entertained 
with illusory knowledge, It is, in fact, a 
call to reason, again to undertake the most 
laborious of all tasks that of self 
examination and to establish a tribunal, 
which may secure it in its well-grounded 
claims, while it pronounces against ...

##Network topology

We have tried various recurrent network topologies. 

The following are some examples of RNNs that either fail to converge or converge too slowly. 

Simple RNN
![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/simple_rnn.png)

Time-skip RNN
![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/rnn_time_skip.png)

Mutually-recurrent RNN
![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/mutually_recurrent_rnn.png)

The one that works reasonably (though quite far from state-of-the-art), is the LSTM (Long-Short Term Memory) network. 

LSTM
![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/lstm.png)

##Learning

The figure shows the learning curve of LSTM on both Shakespeare's and Kant's corpus. 

![enter image description here](https://raw.githubusercontent.com/JimJarvis/DocImages/master/laminar/corpus_learning.png)

The LSTM converges after around 25 epochs. 

Lower perplexity means "easier" language modeling task and thus better learning performance. 

In the figure, the blue curve is trained on Kant's corpus while the red curve is trained on Shakespeare's corpus. Because the blue curve is consistently beneath the red one, it seems that LSTM prefers Kant's text to Shakespeare's for reasons yet unknown. 
