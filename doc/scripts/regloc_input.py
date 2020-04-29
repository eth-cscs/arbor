
############# locsets
ls_root  = {'type': 'locset', 'value': [(0, 0.0)]}
ls_term  = {'type': 'locset', 'value': [(1, 1.0), (3, 1.0), (4, 1.0), (5, 1.0)]}
ls_rand_dend  = {'type': 'locset', 'value': [(0, 0.5547193370156588), (0, 0.5841758202819731), (0, 0.607192003545501), (0, 0.6181091003428546), (0, 0.6190845627201184), (0, 0.7027325639263277), (0, 0.7616129092226993), (0, 0.9645150497869694), (1, 0.15382287505908834), (1, 0.2594719824047551), (1, 0.28087652335178354), (1, 0.3729681478609085), (1, 0.3959560134241004), (1, 0.4629424550242548), (1, 0.47346867377446744), (1, 0.5493486883630476), (1, 0.6227685370674116), (1, 0.6362196581003494), (1, 0.6646511214508091), (1, 0.7157318936458146), (1, 0.7464198558822775), (1, 0.77074507802833), (1, 0.7860238136304932), (1, 0.8988928261704698), (1, 0.9581259332943499), (2, 0.12773985425987294), (2, 0.3365926476076694), (2, 0.44454300804769703), (2, 0.5409466695719178), (2, 0.5767511435223905), (2, 0.6340206909931745), (2, 0.6354772583375223), (2, 0.6807941995943213), (2, 0.774655947503608), (3, 0.05020708596877571), (3, 0.25581431877212274), (3, 0.2958305460715556), (3, 0.296698184761692), (3, 0.509669134988683), (3, 0.7662305637426007), (3, 0.8565839889923518), (3, 0.8889077221517746), (4, 0.24311286693286885), (4, 0.4354361205546333), (4, 0.4467752481260171), (4, 0.5308169153994543), (4, 0.5701465671464049), (4, 0.670081739879954), (4, 0.6995486862583797), (4, 0.8186709628604206), (4, 0.9141224600171143)]}
ls_loc15  = {'type': 'locset', 'value': [(1, 0.5)]}
ls_uniform0  = {'type': 'locset', 'value': [(0, 0.5841758202819731), (1, 0.6362196581003494), (1, 0.7157318936458146), (1, 0.7464198558822775), (2, 0.6340206909931745), (2, 0.6807941995943213), (3, 0.296698184761692), (3, 0.509669134988683), (3, 0.7662305637426007), (4, 0.5701465671464049)]}
ls_uniform1  = {'type': 'locset', 'value': [(0, 0.9778060763285382), (1, 0.19973428495790843), (1, 0.8310607916260988), (2, 0.9210229159315735), (2, 0.9244292525837472), (2, 0.9899772550845479), (3, 0.9924233395972087), (4, 0.3641426305909531), (4, 0.4787812247064867), (4, 0.5138656268861914)]}
ls_branchmid  = {'type': 'locset', 'value': [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (5, 0.5)]}
ls_sample1  = {'type': 'locset', 'value': [(0, 0.3324708796524168)]}
ls_distal  = {'type': 'locset', 'value': [(1, 0.7960259763299439), (3, 0.6666666666666667), (4, 0.39052429175126996), (5, 1.0)]}
ls_proximal  = {'type': 'locset', 'value': [(1, 0.2960259763299439), (2, 0.0), (5, 0.6124999999999999)]}
ls_distint_in  = {'type': 'locset', 'value': [(1, 0.5), (2, 0.7), (5, 0.1)]}
ls_proxint_in  = {'type': 'locset', 'value': [(1, 0.8), (2, 0.3)]}
ls_loctest  = {'type': 'locset', 'value': [(0, 0.0), (1, 1.0)]}

############# regions
reg_empty = {'type': 'region', 'value': []}
reg_all = {'type': 'region', 'value': [(0, 0.0, 1.0), (1, 0.0, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0), (5, 0.0, 1.0)]}
reg_tag1 = {'type': 'region', 'value': [(0, 0.0, 0.3324708796524168)]}
reg_tag2 = {'type': 'region', 'value': [(5, 0.0, 1.0)]}
reg_tag3 = {'type': 'region', 'value': [(0, 0.3324708796524168, 1.0), (1, 0.0, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0)]}
reg_tag4 = {'type': 'region', 'value': []}
reg_soma = {'type': 'region', 'value': [(0, 0.0, 0.3324708796524168)]}
reg_axon = {'type': 'region', 'value': [(5, 0.0, 1.0)]}
reg_dend = {'type': 'region', 'value': [(0, 0.3324708796524168, 1.0), (1, 0.0, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0)]}
reg_radlt5 = {'type': 'region', 'value': [(1, 0.4440389644949158, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0), (5, 0.6562500000000001, 1.0)]}
reg_radlte5 = {'type': 'region', 'value': [(1, 0.4440389644949158, 1.0), (2, 0.0, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0), (5, 0.6562500000000001, 1.0)]}
reg_rad36 = {'type': 'region', 'value': [(1, 0.2960259763299439, 0.7960259763299439), (2, 0.0, 1.0), (3, 0.0, 0.6666666666666667), (4, 0.0, 0.39052429175126996), (5, 0.6124999999999999, 1.0)]}
reg_branch0 = {'type': 'region', 'value': [(0, 0.0, 1.0)]}
reg_branch3 = {'type': 'region', 'value': [(3, 0.0, 1.0)]}
reg_cable_1_01 = {'type': 'region', 'value': [(1, 0.0, 1.0)]}
reg_cable_1_31 = {'type': 'region', 'value': [(1, 0.3, 1.0)]}
reg_cable_1_37 = {'type': 'region', 'value': [(1, 0.3, 0.7)]}
reg_proxint = {'type': 'region', 'value': [(0, 0.7697564611867647, 1.0), (1, 0.4774887508467626, 0.8), (2, 0.0, 0.3)]}
reg_proxintinf = {'type': 'region', 'value': [(0, 0.0, 1.0), (1, 0.0, 0.8), (2, 0.0, 0.3)]}
reg_distint = {'type': 'region', 'value': [(1, 0.5, 0.8225112491532374), (2, 0.7, 1.0), (3, 0.0, 0.432615327328525), (4, 0.0, 0.3628424955125098), (5, 0.1, 0.6)]}
reg_distintinf = {'type': 'region', 'value': [(1, 0.5, 1.0), (2, 0.7, 1.0), (3, 0.0, 1.0), (4, 0.0, 1.0), (5, 0.1, 1.0)]}
reg_lhs = {'type': 'region', 'value': [(0, 0.5, 1.0), (1, 0.0, 0.5)]}
reg_rhs = {'type': 'region', 'value': [(1, 0.0, 1.0)]}
reg_and = {'type': 'region', 'value': [(1, 0.0, 0.5)]}
reg_or = {'type': 'region', 'value': [(0, 0.5, 1.0), (1, 0.0, 1.0)]}
