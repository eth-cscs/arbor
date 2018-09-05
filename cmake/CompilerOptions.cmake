# Compiler-aware compiler options

set(CXXOPT_DEBUG "-g")
set(CXXOPT_CXX11 "-std=c++11")

if(CMAKE_CXX_COMPILER_ID MATCHES "XL")
    # CMake, bless its soul, likes to insert this unsupported flag. Hilarity ensues.
    string(REPLACE "-qhalt=e" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Warning options: disable specific spurious warnings as required.

set(CXXOPT_WALL
    -Wall

    # XL C:
    #
    # * Disable 'missing-braces' warning: this will inappropriately
    #   flag initializations such as
    #       std::array<int,3> a={1,2,3};

    $<IF:$<CXX_COMPILER_ID:XL>,-Wno-missing-braces,>

    # Clang:
    #
    # * Disable 'potentially-evaluated-expression' warning: this warns
    #   on expressions of the form `typeid(expr)` when `expr` has side
    #   effects.

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-potentially-evaluated-expression,>

    # * Clang erroneously warns that T is an 'unused type alias'
    #   in code like this:
    #       struct X {
    #           using T = decltype(expression);
    #           T x;
    #       };

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-unused-local-typedef,>

    # * Ignore warning if string passed to snprintf is not a string literal.

    $<IF:$<CXX_COMPILER_ID:Clang>,-Wno-format-security,>

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

        if(target_model MATCHES "x86" OR target_model MATCHES "amd64")
            set(arch_opt "-march=${arch}")
        else()
            set(arch_opt "-mcpu=${arch}")
        endif()

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        # Translate target architecture names to Intel-compatible names.
        # icc 17 recognizes the following specific microarchitecture names for -mtune:
        #     broadwell, haswell, ivybridge, knl, sandybridge, skylake

        if(arch MATCHES "sandybridge")
            set(tune "${arch}")
            set(arch "AVX")
        elseif(arch MATCHES "ivybridge")
            set(tune "${arch}")
            set(arch "CORE-AVX-I")
        elseif(arch MATCHES "broadwell|haswell|skylake")
            set(tune "${arch}")
            set(arch "CORE-AVX2")
        elseif(arch MATCHES "knl")
            set(tune "${arch}")
            set(arch "MIC-AVX512")
        elseif(arch MATCHES "nehalem|westmere")
            set(tune "corei7")
            set(arch "SSE4.2")
        elseif(arch MATCHES "core2")
            set(tune "core2")
            set(arch "SSSE3")
        elseif(arch MATCHES "native")
            unset(tune)
            set(arch "Host")
        else()
            set(tune "generic")
            set(arch "SSE2") # default for icc
        endif()

        if(tune)
            set(arch_opt "-x${arch};-mtune=${tune}")
        else()
            set(arch_opt "-x${arch}")
        endif()

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "XL")
        # xlC 13 for Linux uses -mcpu. Not even attempting to get xlC 12 for BG/Q right
        # at this point: use CXXFLAGS as required!
        #
        # xlC, gcc, and clang all recognize power8 and power9 as architecture keywords.

        if(arch MATCHES "native")
            set(arch_opt "-qarch=auto")
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
