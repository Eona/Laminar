#Cadenza

An artificial music improviser powered by GPU-based recurrent neural network. 

## Installation

 1. Make sure you have CUDA 7.0 and Nsight eclipse installed.
 
 2. Download [google unit test framework 1.7](https://code.google.com/p/googletest/downloads/detail?name=gtest-1.7.0.zip&can=2&q=). Install the headers to `/opt/gtest/include` and the libraries (*libgtest.a* and *libgtest_main.a*) to `/opt/gtest/lib`
 
 3. Run the script  
  ```
  $ ./gen_eclipse.sh <your_project_dir>
  ``` 
  You should see a `Done` message at the end without any error. 
 
 4. Start nsight eclipse. File -> import project -> navigate to `..../cadenza/<your_project_dir>` -> voila!
 
 5. You might need to manually rebuild the index by right clicking the project -> index -> rebuild. Otherwise eclipse editor will be swamped with error markups. 
Warning: if you ever change any of the `CMakeLists.txt` or add/delete/rename source files, do not build or run directly in eclipse. Run `./gen_eclipse.sh <project>` in a terminal before you do anything in eclipse, otherwise the code indexer will be broken again. 
