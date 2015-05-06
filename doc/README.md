#***Laminar*** Framework

Project *Laminar* aims to provide a comprehensive framework to train and deploy feed-forward neural networks and recurrent neural networks, two of the most important deep learning architectures. 

The name is chosen with a two-fold meaning. In fluid dynamics, the term _Laminar flow_ means a steady flow without turbulence. Deep learning is based on "gradient flow" that propagates through a neural network, so we appropriately steal the concept from physics. 

*LAMINAR* is also a recursive acronym:

__*Laminar Accelerated and MInimalistic Neural ARchitecture*__ 

The framework features:

- Expressive user interface in C++ 11.<br><br>
- Efficient and scalable. The library runs efficiently on heterogenous hardware from multi-threaded CPUs to GPUs. *Laminar* scales as your backend scales. <br><br>
- Versatile. Literally dozens of built-in pluggable modules are shipped with *Laminar*. <br><br>
	- Arbitrarily complicated neural networks can be constructed and trained with ease.
	- Six computational backends are shipped with the library, which support most of the common hardware. <br><br>
- Customizable in every corner from end to end. If the built-in modules do not yet satisfy your needs, you can always roll your own by extending the architecture. <br><br>
- The current code base contains more than **18,800** lines of code. And it is still growing on a daily basis. We plan to release the entire framework to the open-source community in the next few months. <br><br>

This repository holds pre-release versions of *Laminar* documents. Because they have been written under time pressure, some details might be imprecise or even incorrect. We appreciate any comment or bug fix you might have. Your feedback is what makes *Laminar* better. 

To get started, please take a look at our [Tutorial](Tutorial.md). It primarily covers *Laminar* basics and the *User API*. 

For an in-depth discussion of technical details, please refer to the [Manual](Manual.md). It covers both the *User API* in more details, and the *Developer API* if you wish to extend the current architecture. 

Finally, the [Design Document](DesignDoc.md) discusses *Laminar*'s design philosophy and our vision for the future. 