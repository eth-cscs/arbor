#pragma once

#if defined(NMC_HAVE_TBB)
    #include "tbb.hpp"
#elif defined(NMC_HAVE_CTHREAD)
    #include "cthread.hpp"
#else
    #define NMC_HAVE_SERIAL
    #include "serial.hpp"
#endif

