# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\App\JetBrains\CLion 2024.2.3\bin\cmake\win\x64\bin\cmake.exe"

# The command to remove a file.
RM = "D:\App\JetBrains\CLion 2024.2.3\bin\cmake\win\x64\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Common-Operators_CUDA\hellocuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Common-Operators_CUDA\hellocuda\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\hellocuda.dir\depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles\hellocuda.dir\compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles\hellocuda.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\hellocuda.dir\flags.make

CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj: CMakeFiles\hellocuda.dir\flags.make
CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj: D:\Common-Operators_CUDA\hellocuda\1.hellocuda.cu
CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj: CMakeFiles\hellocuda.dir\compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\Common-Operators_CUDA\hellocuda\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/hellocuda.dir/1.hellocuda.cu.obj"
	C:\PROGRA~1\NVIDIA~2\CUDA\v12.6\bin\nvcc.exe -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj -MF CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj.d -x cu -c D:\Common-Operators_CUDA\hellocuda\1.hellocuda.cu -o CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj -Xcompiler=-FdCMakeFiles\hellocuda.dir\,-FS

CMakeFiles\hellocuda.dir\1.hellocuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/hellocuda.dir/1.hellocuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles\hellocuda.dir\1.hellocuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/hellocuda.dir/1.hellocuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target hellocuda
hellocuda_OBJECTS = \
"CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj"

# External object files for target hellocuda
hellocuda_EXTERNAL_OBJECTS =

hellocuda.exe: CMakeFiles\hellocuda.dir\1.hellocuda.cu.obj
hellocuda.exe: CMakeFiles\hellocuda.dir\build.make
hellocuda.exe: CMakeFiles\hellocuda.dir\linkLibs.rsp
hellocuda.exe: CMakeFiles\hellocuda.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=D:\Common-Operators_CUDA\hellocuda\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable hellocuda.exe"
	"D:\App\JetBrains\CLion 2024.2.3\bin\cmake\win\x64\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\hellocuda.dir --rc=D:\WINDOW~2\10\bin\100226~1.0\x64\rc.exe --mt=D:\WINDOW~2\10\bin\100226~1.0\x64\mt.exe --manifests -- D:\App\MICROS~2\2022\PROFES~1\VC\Tools\MSVC\1441~1.341\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\hellocuda.dir\objects1.rsp @<<
 /out:hellocuda.exe /implib:hellocuda.lib /pdb:D:\Common-Operators_CUDA\hellocuda\cmake-build-debug\hellocuda.pdb /version:0.0 /machine:x64 /debug /INCREMENTAL @CMakeFiles\hellocuda.dir\linkLibs.rsp -LIBPATH:"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64" 
<<

# Rule to build all files generated by this target.
CMakeFiles\hellocuda.dir\build: hellocuda.exe
.PHONY : CMakeFiles\hellocuda.dir\build

CMakeFiles\hellocuda.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\hellocuda.dir\cmake_clean.cmake
.PHONY : CMakeFiles\hellocuda.dir\clean

CMakeFiles\hellocuda.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" D:\Common-Operators_CUDA\hellocuda D:\Common-Operators_CUDA\hellocuda D:\Common-Operators_CUDA\hellocuda\cmake-build-debug D:\Common-Operators_CUDA\hellocuda\cmake-build-debug D:\Common-Operators_CUDA\hellocuda\cmake-build-debug\CMakeFiles\hellocuda.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles\hellocuda.dir\depend

