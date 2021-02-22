include(CheckCXXSourceCompiles)

# Compiler-aware compiler options
set(CXXOPT_DEBUG "-g")
set(CXXOPT_CXX11 "-std=c++11")

if(CMAKE_CXX_COMPILER_ID MATCHES "XL")
    # CMake, bless its soul, likes to insert this unsupported flag. Hilarity ensues.
    string(REPLACE "-qhalt=e" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()


if(${ARBDEV_COLOR})
    set(colorflags
        $<IF:$<CXX_COMPILER_ID:Clang>,-fcolor-diagnostics,>
        $<IF:$<CXX_COMPILER_ID:AppleClang>,-fcolor-diagnostics,>
        $<IF:$<CXX_COMPILER_ID:GNU>,-fdiagnostics-color=always,>)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${colorflags}>")
endif()

# A library to collect compiler-specific linking adjustments.
add_library(arbor-compiler-compat INTERFACE)

# Check how to use std::filesystem
string(CONFIGURE [[
  #include <cstdlib>
  #include <filesystem>
  int main() {
    auto cwd = std::filesystem::current_path();
    printf("%s", cwd.c_str());
  }
  ]] arb_cxx_fs_test @ONLY)

set(STD_FS_LIB "")
set(CMAKE_REQUIRED_FLAGS -std=c++17)
check_cxx_source_compiles("${arb_cxx_fs_test}" STD_FS_PLAIN)

if(NOT STD_FS_PLAIN)
  set(STD_FS_LIB -lstdc++fs)
  set(CMAKE_REQUIRED_FLAGS -std=c++17)
  set(CMAKE_REQUIRED_LIBRARIES ${STD_FS_LIB})
  check_cxx_source_compiles("${arb_cxx_fs_test}" STD_FS_STDCXX)

  if(NOT STD_FS_STDCXX)
    set(STD_FS_LIB -lc++fs)
    set(CMAKE_REQUIRED_FLAGS -std=c++17)
    set(CMAKE_REQUIRED_LIBRARIES ${STD_FS_LIB})
    check_cxx_source_compiles("${arb_cxx_fs_test}" STD_FS_CXX)

    if(NOT STD_FS_CXX)
      message(FATAL_ERROR "Could not enable support for std::filesystem")
    endif()
  endif()
endif()

target_link_libraries(arbor-compiler-compat INTERFACE ${STD_FS_LIB})

install(TARGETS arbor-compiler-compat EXPORT arbor-targets)

# Warning options: disable specific spurious warnings as required.

set(CXXOPT_WALL
    -Wall

    # Clang:
    #
    # * Disable 'missing-braces' warning: this will inappropriately
    #   flag initializations such as
    #       std::array<int,3> a={1,2,3};

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-missing-braces,>
    $<IF:$<CXX_COMPILER_ID:AppleClang>,-Wno-missing-braces,>

    # Clang:
    #
    # * Disable 'potentially-evaluated-expression' warning: this warns
    #   on expressions of the form `typeid(expr)` when `expr` has side
    #   effects.

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-potentially-evaluated-expression,>
    $<IF:$<CXX_COMPILER_ID:AppleClang>,-Wno-potentially-evaluated-expression,>

    # Clang (Apple):
    #
    # * Disable 'range-loop-analysis' warning: disabled by default in
    #   clang, but enabled in Apple clang, this will flag loops of the form
    #   `for (auto& x: y)` where iterators for `y` dereference to proxy objects.
    #   Such code is correct, and the warning is spurious.

    $<IF:$<CXX_COMPILER_ID:AppleClang>,-Wno-range-loop-analysis,>

    # Clang:
    #
    # * Disable 'unused-function' warning: this will flag the unused
    #   functions generated by `ARB_DEFINE_LEXICOGRAPHIC_ORDERING`

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-unused-function,>
    $<IF:$<CXX_COMPILER_ID:AppleClang>,-Wno-unused-function,>

    # * Clang erroneously warns that T is an 'unused type alias'
    #   in code like this:
    #       struct X {
    #           using T = decltype(expression);
    #           T x;
    #       };

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-unused-local-typedef,>
    $<IF:$<CXX_COMPILER_ID:AppleClang>,-Wno-unused-local-typedef,>

    # * Ignore warning if string passed to snprintf is not a string literal.

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-format-security,>
    $<IF:$<CXX_COMPILER_ID:AppleClang>,-Wno-format-security,>

    # GCC:
    #
    # * Disable 'maybe-uninitialized' warning: this will be raised
    #   inappropriately in some uses of util::optional<T> when T
    #   is a primitive type.

    $<IF:$<CXX_COMPILER_ID:GNU>,-Wno-maybe-uninitialized,>

    # Intel:
    #
    # Disable warning for unused template parameter
    # this is raised by a templated function in the json library.

    $<IF:$<CXX_COMPILER_ID:Intel>,-wd488,>)


# Set ${optvar} in parent scope according to requested architecture.
# Architectures are given by the same names that GCC uses for its
# -mcpu or -march options.

function(set_arch_target optvar arch)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        # Correct compiler option unfortunately depends upon the target architecture family.
        # Extract this information from running the configured compiler with --verbose.

        try_compile(ignore ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/dummy.cpp COMPILE_DEFINITIONS --verbose OUTPUT_VARIABLE cc_out)
        string(REPLACE "\n" ";" cc_out "${cc_out}")
        set(target)
        foreach(line ${cc_out})
            if(line MATCHES "^Target:")
                string(REGEX REPLACE "^Target: " "" target "${line}")
            endif()
        endforeach(line)
        string(REGEX REPLACE "-.*" "" target_model "${target}")

        # Use -mcpu for all supported targets _except_ for x86, where it should be -march.

        if(target_model MATCHES "x86|i[3456]86" OR target_model MATCHES "amd64" OR target_model MATCHES "aarch64")
            set(arch_opt "-march=${arch}")
        else()
            set(arch_opt "-mcpu=${arch}")
        endif()
    endif()

    get_property(enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if ("CUDA" IN_LIST enabled_languages)
        # Prefix architecture options with `-Xcompiler=` when compiling CUDA sources, i.e.
        # with nvcc.
        set(arch_opt_cuda_guarded)
        foreach(opt ${arch_opt})
            list(APPEND arch_opt_cuda_guarded "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>${opt}")
        endforeach()

        set("${optvar}" "${arch_opt_cuda_guarded}" PARENT_SCOPE)
    else()
        set("${optvar}" "${arch_opt}" PARENT_SCOPE)
    endif()

endfunction()
