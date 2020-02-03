:COMMENT
:
:   **************************************************
:   File generated by: neuroConstruct v1.3.7
:   **************************************************
:
:
:ENDCOMMENT
:
:
: This is a NEURON mod file generated from a ChannelML file
:
:  Unit system of original ChannelML file: Physiological Units
:
:COMMENT
:    ChannelML file containing a single Channel description
:ENDCOMMENT
:
:TITLE Channel: Leak
:
:COMMENT
:    Simple example of a leak/passive conductance. Note: for GENESIS cells with a single leak conductance,
:        it is better to use the Rm and Em variables for a passive current.
:ENDCOMMENT


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
    (um) = (micrometer)
    (molar) = (1/liter)
    (mM) = (millimolar)
    (l)  = (liter)
}


    
NEURON {
    SUFFIX Leak
    : A non specific current is present
    RANGE e
    NONSPECIFIC_CURRENT il
:    RANGE gmax, gion, il
    RANGE gmax, il
}

PARAMETER { 
    gmax = 0.0003 (S/cm2) : default value, should be overwritten when conductance placed on cell
    e    = -80    (mV)    : default value, should be overwritten when conductance placed on cell
}



ASSIGNED {
    v (mV)
:    il (mA/cm2) :neuron
}

BREAKPOINT { 
    il = gmax*(v - e)
}
