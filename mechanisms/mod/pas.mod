UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

NEURON {
    SUFFIX pas
    NONSPECIFIC_CURRENT i
    RANGE e
}

STATE {
    g
}

INITIAL {}

PARAMETER {
    e = -65  (mV) : we use -65 for the ball and stick model, instead of Neuron default of -70
}

ASSIGNED {
    v (mV)
}

BREAKPOINT {
    i = g*(v - e)
}
