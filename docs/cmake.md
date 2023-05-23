
### manual compilation
    g++ main.cpp addition.hpp division.hpp print_result.hpp
    The -c option instructs to only preprocess, compile, and assemble steps. <no-linking>
    This is compilation: Don't care about .cpp files now.
    g++ -c main.cpp addition.hpp division.hpp print_result.hpp



### Some options:
    -g                  turn on debugging (so GDB gives more friendly output)
    -Wall               turns on most warnings
    -O or -O2           turn on optimizations
    -o <name>           name of the output file
    -c                  output an object file (.o)
    -I<include path>    specify an include directory
    -L<library path>    specify a lib directory
    -l<library>         link with library lib<library>.a


### cmake commands docs

    @ cmake_minimum_required(VERSION <cmake_version>)  
        { specify what minimum version is required }

    @ project(<project_name> VERSION <project_version>)  
        { specify project name and version }
    
    @ add_executable(<executable-name>, <source_file1.cpp, ...>)
        { generate the executable from the source files }

    @ add_library(<lib_name> <lib_source_files.cpp, ...>)
        { create a library }
    
    @ target_link_libraries(<target>
        { for linking a library to executables }
                    <PRIVATE|PUBLIC|INTERFACE> <item>...
                    [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)


#### some notes on targets:

- The libraries and the executables are jointly called `targets`.  
- `targets` have some `Properties` & `Dependencies` associated with it.  
- very common `target` Prperties:

        INTERFACE_LINK_DIRECTORIES
        INCLUDE_DIRECTORIES
        VERSION
        SOURCES
- The properties can also be modified/retrieved using commands like:
    
        set_target_property()
        get_target_property()
        set_property()
        get_property()
- If `target B` is dependency of `target A`,  
then `target A` can only be built after `target B` is built successfullt.


#### cmake commands docs continued

    @ add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL] [SYSTEM])
        { source_dir is where  the source CMakeLists.txt and code files are located }
        { binary_dir is where the output files will be put }

    @ target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
        <INTERFACE|PUBLIC|PRIVATE> [items1...]
        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
    -- or --
    @ target_include_directories(<target>
                <scope>
                <dir1>
                ...)
        { Specifies include directories to use when compiling a given target }
        { The named <target> must have been created by a command such as `add_executable()` or `add_library()` }
        { This command can be used only after adding the targets have been created }


#### some notes on target properties:

- target_include_directories(\<target\> \<scope\> \<dir\>)  
        ex: `target_include_directories(target-name PUBLIC include)`

    Property being set here: `INTERFACE_INCLUDE_DIRECTORIES`  
    Of target: `target-name`  
    Property set to: `PUBLIC`  

- How to decide when to use `PUBLIC` `PRIVATE` or `INTERFACE` scopes.  
    When setting scope of a directory for a target, answer the following:  
    |    scope to use                               |PUBLIC |INTERFACE  |PRIVATE|
    |-----------------------------------------------|-------|-----------|-------|
    |   Does the `target` need the dir?             |   YES |   NO      | YES   |
    |   Do other targets, depending upon `target`   |   YES |   YES     | NO    |
    |       need this dir?                          |       |           |       |

-   |Command                        |   Property set by the command     |
    |-------------------------------|-----------------------------------|
    |@ target_compile_definitions   | INTERFACE_COMPILE_DEFINITIONS     |
    |@ target_sources               | INTERFACE_SOURCES                 |
    |@ target_compile_features      | INTERFACE_COMPILE_FEATURES        |
    |@ target_compile_options       | INTERFACE_COMPILE_OPTIONS         |
    |@ target_link_directories      | INTTERFACE_LINK_DIRECTORIES       |
    |@ target_link_libraries        | INTERFACE_LINK_LIBRARIES          |
    |@ target_link_options          | INTERFACE_LINK_OPTIONS            |
    |@ target_precompile_headers    | INTERFACE_PRECOMPILE_HEADERS      |

    These commands in general need `command(<target-name> <scope-specifier> <args>)`
  
  
### sample directory structure and CMakeLists.txt

    .                                           |---------------- proj-root/CMakeLists.txt -----------------|
    ├── CMakeLists.txt -----------------------> | cmake_minimum_required(VERSION 3.0.0)                     |
    ├── build                                   | project(calculator-project VERSION 1.0.0)                 |
    ├── main.cpp                                | add_subdirectory(my_print)                                |
    ├── my_math                                 | add_subdirectory(my_math)                                 |
    │   ├── CMakeLists.txt -----------------|   |                                                           |
    │   ├── include                         |   | add_executable(calc             # executable-target       |
    │   │   └── my_math                     |   |             main.cpp)           # executable-source-file  |
    │   │       ├── addition.hpp            |   |                                                           |
    │   │       └── division.hpp            |   | target_link_libraries(calc      # executable-target       |
    │   └── src                             |   |     PRIVATE                     # scope-specifier         |
    │       ├── addition.cpp                |   |     my_math                     # library-target          |
    │       └── division.cpp                |   |     my_print)                   # library-target          |
    └── my_print                            |   |-----------------------------------------------------------|
        ├── CMakeLists.txt -------------|   |
        ├── include                     |   |       |------------ proj-root/my_math/CMakeLists.txt -------------|
        │   └── my_print                |   |-----> |   add_library(my_math             # executable-target     |
        │       └── print_result.hpp    |           |               src/addition.cpp    # library-source-file   |
        └── src                         |           |               src/division.cpp)   # library-source-file   |
            └── print_result.cpp        |           |   target_include_directories(my_math PUBLIC include)      |
                                        |           |-----------------------------------------------------------|
                                        |    
                                        |       |------------- proj-root/my_print/CMakeLists.txt ---------------|
                                        |---->  |   add_library(my_print                 # executable-target    |
                                                |               src/print_result.cpp)   # library-source-file   |
                                                |   target_include_directories(my_print PUBLIC include)         |    
                                                |---------------------------------------------------------------|
    
    
### get variable from a commad + print variable + 
```
    execute_process(
        COMMAND python -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message("Pybind11 Found at: ${PYBIND11_PATH}")
    set(pybind11_DIR ${PYBIND11_PATH})
```