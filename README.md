#Project LAMINAR

**Laminar Accelerated and MInimalistic Neural ARchitecture**

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

Finally, the [Design Document](DesignDoc.md) discusses *Laminar*'s design philosophy and our vision for the future. '


## Installation

 1. Install CMake, the minimum version is 2.8.
 
 2. Make sure you have CUDA 7.0 and Nsight eclipse installed to the default location. Clic CUDA can be found at `/usr/local/cuda-7.0`.
 
 3. Download [google unit test framework 1.7](https://code.google.com/p/googletest/downloads/detail?name=gtest-1.7.0.zip&can=2&q=). Install the headers to `/opt/gtest/include` and the libraries (*libgtest.a* and *libgtest_main.a*) to `/opt/gtest/lib`
 
 4. Run the script  
  ```
  $ ./gen_eclipse.sh <your_project_dir>
  ``` 
  You should see a `Done` message at the end without any error. 
 
 5. Start nsight eclipse. File -> import project -> navigate to `..../cadenza/<your_project_dir>` -> voila!
 
 6. You might need to manually rebuild the index by right clicking the project -> index -> rebuild. Otherwise eclipse editor will be swamped with error markups. 
Warning: if you ever change any of the `CMakeLists.txt` or add/delete/rename source files, do not build or run directly in eclipse. Run `./gen_eclipse.sh <project>` in a terminal before you do anything in eclipse, otherwise the code indexer will be broken again. 

