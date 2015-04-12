#Project LAMINAR

**Laminar Accelerated and MInimalistic Neural ARchitecture**

A generic deep learning framework powered by drop-in Laminar virtual engines. 

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

## Notes
### Naming convention

 - Functions use underscores
 - Macro functions use all-CAPS and underscores
 - Variables start with lower case and use camel case
 - Types and classes start with upper case and use camel case
