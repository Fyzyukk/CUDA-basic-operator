# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/build

# Include any dependencies generated for this target.
include CMakeFiles/reduce_v8.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reduce_v8.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reduce_v8.dir/flags.make

CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.o: CMakeFiles/reduce_v8.dir/flags.make
CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.o: ../reduce_v8_half.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.o"
	/usr/local/cuda-11.8/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/reduce_v8_half.cu -o CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.o

CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target reduce_v8
reduce_v8_OBJECTS = \
"CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.o"

# External object files for target reduce_v8
reduce_v8_EXTERNAL_OBJECTS =

app/reduce_v8: CMakeFiles/reduce_v8.dir/reduce_v8_half.cu.o
app/reduce_v8: CMakeFiles/reduce_v8.dir/build.make
app/reduce_v8: CMakeFiles/reduce_v8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable app/reduce_v8"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduce_v8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reduce_v8.dir/build: app/reduce_v8

.PHONY : CMakeFiles/reduce_v8.dir/build

CMakeFiles/reduce_v8.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reduce_v8.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reduce_v8.dir/clean

CMakeFiles/reduce_v8.dir/depend:
	cd /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/build /home/yst/文档/yst/CUDA/cuda-mode/CUDA-Optimization/Reduce/build/CMakeFiles/reduce_v8.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reduce_v8.dir/depend

