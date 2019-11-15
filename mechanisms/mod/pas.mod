UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

NEURON {
    SUFFIX pas
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

INITIAL {}

PARAMETER {
    g = .001 (S/cm2)
    e = -65  (mV) : we use -65 for the ball and stick model, instead of Neuron default of -70
}

ASSIGNED {
    v (mV)
}

BREAKPOINT {
    if (g > 2) {
        if (g > 3) {
            i = 0
        } else {
            i = 1
        }
    } else {
        if (g < 1) {
            i = 2
        } else {
            i = 3
        }
    }
:     i = g*(v-e)
}
