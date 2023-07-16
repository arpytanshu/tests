Setting up libtorch and building example-app from pytorch docs.

Used this cmake file referenced from this blog: ```https://www.neuralception.com/settingupopencv/```

### get libtorch zip
```
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.13.0.zip
unzip libtorch-macos-1.13.0.zip -d /opt/
```


### CMakeLists.txt
```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)
set(Torch_DIR /opt/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_OSX_ARCHITECTURES "x86_64")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
```

### build commands
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
cmake --build . --config Release
```