import numpy as np
import math
from numba import njit, prange

pi = math.pi
cos = math.cos
exp = math.exp
sqrt = math.sqrt
pow = math.pow

# Set 0d model constants
# R-resistance, Y(L)-inertance, E-elastance, C-compliance
# Z-volume coefficent, S-Viscoelastic coefficent

# Cardaiac chamber Elastance (left/right ventricle and atrium) 心室、心房
Elva=2.87      # Peak-systolic elastance of left ventricle # 2.75?
Elvb=0.06      # Basic diastolic elastance of left ventricle # 0.08?
Elaa=0.07      # Peak-systolic elastance of left atrium 
Elab=0.075     # Basic diastolic elastance of left atrium # 0.09?
Erva=0.52      # Peak-systolic elastance of right ventricle # 0.55?
Ervb=0.043     # Basic diastolic elastance of right ventricle # 0.05?
Eraa=0.055     # Peak-systolic elastance of right atrium # 0.06?
Erab=0.06      # Basic diastolic elastance of right atrium # 0.07?

Vmax=900.0  # Reference volume of Frank-Starling law
Es=45.9     # Effective septal elastance
Vpc0=380.0  # Reference total pericardial and cardiac volume
Vpe=30.0    # Pericardial volume of heart
Vcon=40.0   # Volume constant
Sva0=0.0005 # Coefficient of cardiac viscoelasticity
 
# Cardiac valve parameters 心臓弁
# (aortic valve(AV),mitral valve(MV), tricuspid valve(TV),pulmonary valve(PV))      
bav=0.000025*0.5 # Bernoulli's resistance of AV # 0.000025?
bmv=0.000016 # Bernoulli's resistance of MV
btv=0.000016 # Bernoulli's resistance of TV
bpv=0.000025*0.5 # Bernoulli's resistance of PV # 0.000025?
Rav=0.005*0.5    # Viscous resistance of AV
Rmv=0.005    # Viscous resistance of MV # 0.001?
Rtv=0.005    # Viscous resistance of TV # 0.001?
Rpv=0.005*0.5    # Viscous resistance of PV
yav=0.0005   # Inertance of AV
ymv=0.0002   # Inertance of MV
ytv=0.0002   # Inertance of TV
ypv=0.0005   # Inertance of PV   
invyav=1.0/yav
invymv=1.0/ymv
invytv=1.0/ytv
invypv=1.0/ypv

# Elastance of pulmatory circulation
Epua0=0.02
Epuc0=0.02
Epuv0=0.02
Epwc0=0.7
Epwv0=0.7

# Vena cava (when 0D is run separately) 大静脈, not used
# Rv=0.0125
# Zv=100.0     
# SSv=0.01
# Rvc=0.0055
# yvc=0.0005
# Zvc=80.0
# Svc=0.01

# Aorta 大動脈
Eaa0=1.0 # used
Raa=0.00196 # not used in 0D model # 0.04?
yaa=0.0005 # not used in 0D model
Saa=0.01 # used
Zaa=1.0 # used
# Ra=1.2 # used but only when ncoup=1
#Za=70.0 # not used in 0D model
#Sa=0.01 # not used in 0D model
#yda=0.00160 # not used in 0D model
#Zda=20.0 # not used in 0D model
#Sda=0.01 # not used in 0D model

# Z: correspond to V node
Zpua=20.0 # can't find reference
Zpuc=60.0
Zpuv=200.0
Zpwa=1.0 # not used
Zpwc=1.0
Zpwv=1.0

# C: correspond to V node
C_upper=0.05 # 0.03?
C_upperc=1.0 # 0.5?
C_upperv=15.0
C_vcuu=5.0
C_vcu=2.0 # no ref?
C_lower=0.2 # 0.1?
C_lowerc=3.0 # 1.5?
C_lowerv=75.0
C_vclu=15.0
C_vcl=3.0 # no ref?
# inverse of C: correspond to V node
invC_upper=1.0/C_upper
invC_upperc=1.0/C_upperc
invC_upperv=1.0/C_upperv
invC_vcuu=1.0/C_vcuu
invC_vcu=1.0/C_vcu
invC_lower=1.0/C_lower
invC_lowerc=1.0/C_lowerc
invC_lowerv=1.0/C_lowerv
invC_vclu=1.0/C_vclu
invC_vcl=1.0/C_vcl

# R: correspond to Q node
Rpua=0.04
Rpuc=0.04
Rpuv=0.005
Rpwa=0.0005 # not used
Rpwc=0.4
Rpwv=0.4
Rup_t  = 0.650231069667257 # don't know how this is derived
R_upperc  = 1.0/Rup_t*0.3/0.7 # 0.6591 <- 0.97?
R_upperv  = R_upperc*0.15 # 0.0988655 <- 0.14?
R_vcuu    = R_upperv*0.2 # 0.0197731 <- 0.03?
R_vcu     = R_vcuu*0.03 # 0.000593193 <- 0.0005?
R_vcu1=0.0005 # <- 0.0005
Rlow_t  = 1.48499254007376 # don't know how this is derived
R_lowerc  = 1.0/Rlow_t*0.3/0.7 # 0.2883 <- 0.29
R_lowerv  = R_lowerc*0.15 # 0.043245 <- 0.04
R_vclu    = R_lowerv*0.2 # 0.008649 <- 0.009
R_vcl     = R_vclu*0.03 # 0.00025947 <- 0.0005?
R_vcl1=0.0005 # <- 0.0005

# y(L): correspond to Q node
ypua=0.0005
ypuc=0.0005
ypuv=0.0005
ypwa=0.0005
ypwc=0.0005
ypwv=0.0005
#y_upper=0.005
y_upperc=0.002 # 0.003?
y_upperv=0.001
y_vcuu=0.0005
y_vcu=0.0005
y_vcu1=0.0005
#y_lower=0.005
y_lowerc=0.002 # 0.003?
y_lowerv=0.001
y_vclu=0.0005
y_vcl=0.0005
y_vcl1=0.0005
# inverse of y(L): correspond to Q node
invypua=1.0/ypua
invypuc=1.0/ypuc
invypuv=1.0/ypuv
invypwa=1.0/ypwa
invypwc=1.0/ypwc
invypwv=1.0/ypwv
invy_upperc=1.0/y_upperc
invy_upperv=1.0/y_upperv
invy_vcuu=1.0/y_vcuu
invy_vcu=1.0/y_vcu
invy_vcu1=1.0/y_vcu1
invy_lowerc=1.0/y_lowerc
invy_lowerv=1.0/y_lowerv
invy_vclu=1.0/y_vclu
invy_vcl=1.0/y_vcl
invy_vcl1=1.0/y_vcl1

# S: correspond to V node
Spua=0.01
Spuc=0.01
Spuv=0.01
Spwa=0.01
Spwc=0.01
Spwv=0.01
S_upper=0.01
S_upperc=0.01
S_upperv=0.01
S_vcuu=0.01
S_vcu=0.01
#S_vcu1=0.01
S_lower=0.01
S_lowerc=0.01
S_lowerv=0.01
S_vclu=0.01
S_vcl=0.01
#S_vcl1=0.01

# other parameters
gpw = 0.0 # 0: no graft, other: graft
pit = -2.5

@njit(fastmath=True)
def update_cat1(Cval, tcr, tee, tar, tac, Tduration, v, result):

    if tcr == 0.0:
        Cval[0] = 1.0 - result[21, 0] / Vmax # FL
        Cval[1] = 1.0 - result[7, 0] / Vmax # FR1

    FL = Cval[0]
    FR1 = Cval[1]

    Cval[2] = lv(tcr, tee, FL) # elv
    Cval[3] = la(tcr, tar, tac, Tduration) # ela
    Cval[4] = rv(tcr, tee, FR1) # erv
    Cval[5] = ra(tcr, tar, tac, Tduration) # era

    elv = Cval[2]
    ela = Cval[3]
    erv = Cval[4]
    era = Cval[5]

    cklr = erv / (Es + erv)
    ckrl = elv / (Es + elv)

    denom_l = 1.0 - cklr
    denom_r = 1.0 - ckrl
    Cval[6] = ((ckrl * Es * v[6]) + (ckrl * cklr * Es * v[1])) / denom_l if denom_l != 0 else 0.0 # plv
    Cval[7] = ((cklr * Es * v[1]) + (ckrl * cklr * Es * v[6])) / denom_r if denom_r != 0 else 0.0 # prv

    plv = Cval[6]
    prv = Cval[7]

    Cval[8] = Sva0 * v[5] * ela # sla
    Cval[9] = Sva0 * plv # slv
    Cval[10] = Sva0 * v[0] * era # sra
    Cval[11] = Sva0 * prv # srv

    inv_Vcon = 1.0 / Vcon
    ppp = (v[0] + v[1] + v[5] + v[6] + Vpe - Vpc0) * inv_Vcon
    Cval[12] = exp(ppp) # ppc

    return Cval

@njit(fastmath=True)
def update_eval(Eval, v):

    Eval[0] = ecal(Eaa0, Zaa, v[7]) # Eaa
    Eval[1] = ecal(Epua0, Zpua, v[2]) # Epua
    Eval[2] = ecal(Epuc0, Zpuc, v[3]) # Epuc
    Eval[3] = ecal(Epuv0, Zpuv, v[4]) # Epuv
    Eval[4] = ecal(Epwc0, Zpwc, v[18]) # Epwc
    Eval[5] = ecal(Epwv0, Zpwv, v[19]) # Epwv

    return Eval

@njit(fastmath=True)
def update_valve(Avalve, Eval, Cval, q, v, dv):

    ela = Cval[3]
    era = Cval[5]
    plv = Cval[6]
    prv = Cval[7]
    slv = Cval[9]
    ppc = Cval[12]
    Eaa = Eval[0]
    Epua = Eval[1]

    # amv
    discriminant = ela * v[5] - plv - Rmv * q[5] - bmv * q[5] * abs(q[5])
    if discriminant > 0.0:     
        Avalve[1] = 4.0
    else:
        Avalve[1] = 0.0

    # aav
    discriminant = plv + slv * dv[6] + ppc - Eaa * Zaa - Rav * q[6] - bav * q[6] * abs(q[6])
    if discriminant > 0.0:     
        Avalve[0] = 4.0
    else:
        Avalve[0] = 0.0

    # apv
    discriminant = prv - Epua * Zpua - Rpv * q[1] - bpv * q[1] * abs(q[1])
    if discriminant > 0.0:     
        Avalve[3] = 4.0
    else:
        Avalve[3] = 0.0

    # atv
    discriminant = era * v[0] - prv - Rtv * q[0] - btv * q[0] * abs(q[0])
    if discriminant > 0.0:     
        Avalve[2] = 4.0
    else:
        Avalve[2] = 0.0

    return Avalve

# Correspondence array between 0d node number and terminal artery number
# row1: 1d artery number, row2: 0d node number(q:n,v:n+1), row3: upper=0/lower=1/brain=2/lma=3
arterynode = np.array([
    [58, 40, 2],
    [61, 42, 2],
    [63, 44, 2],
    [65, 46, 2],
    [67, 48, 2],
    [70, 50, 2],
    [74, 52, 0],
    [75, 54, 0],
    [78, 56, 0],
    [79, 58, 0],
    [80, 60, 0],
    [81, 62, 0],
    [82, 64, 0],
    [83, 66, 0],
    [14, 68, 1],
    [26, 70, 1],
    [28, 72, 1],
    [30, 74, 1],
    [32, 76, 1],
    [8 , 78, 0],
    [19, 80, 0],
    [22, 82, 1],
    [43, 84, 0],
    [44, 86, 0],
    [45, 88, 0],
    [46, 90, 0],
    [23, 92, 1],
    [24, 94, 1],
    [36, 96, 1],
    [51, 98, 1],
    [37,100, 1],
    [52,102, 1],
    [41,104, 1],
    [42,106, 1],
    [54,108, 1],
    [55,110, 1],
    [84,112, 3],
    [85,113, 3],
    [86,114, 3],
    [87,115, 3],
    [88,116, 3],
    [89,117, 3],
    [90,118, 3],
    [91,119, 3],
    [92,121, 3],
    [93,123, 3],
    [94,125, 3],
    [95,127, 3],
    [96,129, 3]]
    ,dtype=int)

# for i_upper_or_lower
iupper = [74,75,78,79,80,81,82,83,8,19,43,44,45,46]
ilower = [14,26,28,30,32,22,23,24,36,51,37,52,41,42,54,55]
icelebral = [58,61,63,65,67,70]

LMA_connection = np.array([ # LMA connection arteries
    [58, 63], # 84
    [61, 58], # 85
    [63, 61], # 86
    [65, 63], # 87
    [67, 65], # 88
    [70, 67], # 89
    [65, 70], # 90
], dtype=np.int64)

@njit
def lookup_value(array, search_col, search_val, return_col):
    # search_col: column to search (0 or 1)
    # search_val: value to search for
    # return_col: column to return (0 or 1)
    matches = array[array[:, search_col] == search_val]
    if matches.size == 0:
        return 0  # 見つからない場合
    return matches[0, return_col]

@njit(fastmath=True)   
def ecal(E, Z, vol):

    Ecal = E * exp(vol / Z)

    return Ecal

@njit(fastmath=True)
def lv(tcr,tee,FL): # left ventricle: elv
    elv = 0.0
    if tcr <= tee:
        elv = FL * Elva * 0.5 * (1.0 - cos(pi * tcr / tee)) + Elvb / FL
    elif tcr <= 1.5 * tee:
        elv = FL * Elva * 0.5 * (1.0 + cos(pi * (tcr - tee) / (0.5 * tee))) + Elvb / FL
    else:
        elv = Elvb / FL

    return elv

@njit(fastmath=True)    
def la(tcr,tar,tac,Tduration): # left atrium: ela
    teec = tar - tac
    teer = teec
    tap = tar + teec - Tduration
    ela = 0.0
    if tcr >= 0.0 and tcr <= tap:
        ela = Elaa * 0.5 * (1.0 + cos(pi * (tcr + Tduration - tar) / teer)) + Elab
    elif tcr > tap and tcr <= tac:
        ela = Elab 
    elif tcr > tac and tcr <= tar:
        ela = Elaa * 0.5 * (1.0 - cos(pi * (tcr - tac) / teec)) + Elab
    elif tcr > tar and tcr <= Tduration:
        ela = Elaa * 0.5 * (1.0 + cos(pi * (tcr - tar) / teer)) + Elab

    return ela

@njit(fastmath=True)
def rv(tcr,tee,FR1): # right ventricle: erv
    erv = 0.0
    if tcr <= tee:
        erv = FR1 * Erva * 0.5 * (1.0 - cos(pi * tcr / tee)) + Ervb / FR1
    elif tcr <= 1.5 * tee:
        erv = FR1 * Erva * 0.5 * (1.0 + cos(2.0 * pi * (tcr - tee) / tee)) + Ervb / FR1 # changed tee-> 0.5 * tee
    else:
        erv = Ervb / FR1

    return erv
    
@njit(fastmath=True)
def ra(tcr,tar,tac,Tduration): # right atrium: era
    teec = tar - tac
    teer = teec
    tap = tar + teer - Tduration
    era = 0.0
    if tcr >= 0.0 and tcr <= tap:
        era = Eraa * 0.5 * (1.0 + cos(pi * (tcr + Tduration - tar) / teer)) + Erab
    elif tcr > tap and tcr <= tac:
        era = Erab 
    elif tcr > tac and tcr <= tar:
        era = Eraa * 0.5 * (1.0 - cos(pi * (tcr - tac) / teec)) + Erab
    elif tcr > tar and tcr <= Tduration:
        era = Eraa * 0.5 * (1.0 + cos(pi * (tcr - tar) / teer)) + Erab

    return era

# function for simulation loop
# 0-1d coupling computation
@njit(fastmath=True)
def interface_01d(nnn, nartery, imaxtree, Atreem, Qtreem, Utreem, Qtree, Atree,
                  A0, P1u, p0, mbif_par, Qguess, nzeromodel, Dif_ratio, Q_temp, A_temp, W1, W2, rukuk,
                  Vtree0d, Ptree0d, Qtree0d, v, dv, q, dvq, P_0d, result, tee, tac, tar, tcr,
                  dx, dt, roi, tbeta, A_term1, remuda0, c_relax, itermax_0d, Tduration, nbegin,
                  iterrelax_0d, converge_cri_coup, mmhgtoPa, RLCtree, numlma, nzm_lma, nzm_cowoutlet, Cval, Eval,
                  Avalve,
                  arterynode_map, i_upper_or_lower, artery_to_0dindex, sim_LMA, disregard_LMA):
    # Initialization
    if nnn == nbegin:
        print("Initialize 0d model calculation.")

    Qguess, W1, W2, A_term1, remuda0 = Qguesscal(nartery, mbif_par, imaxtree, Atree, Qtree, A0, Qtreem,
                                                 dt, roi, dx, Qguess, W1, W2, tbeta, A_term1, remuda0)

    niter_0dcount = 0
    C_ratto = 0.0

    result[:, 0] = result[:, 1]

    while niter_0dcount < itermax_0d + 1:

        result, P1u, q, v, dv, dvq, P_0d, Ptree0d, Qtree0d, Qguess, Vtree0d, Cval, Eval, Avalve = zerodmodel(nnn,
                                                                                                             nartery,
                                                                                                             mbif_par,
                                                                                                             niter_0dcount,
                                                                                                             nzeromodel,
                                                                                                             Qguess,
                                                                                                             result,
                                                                                                             tee, tac,
                                                                                                             tar, tcr,
                                                                                                             q, v, dv,
                                                                                                             dvq, rukuk,
                                                                                                             P_0d,
                                                                                                             Ptree0d,
                                                                                                             Qtree0d,
                                                                                                             Vtree0d,
                                                                                                             P1u,
                                                                                                             RLCtree,
                                                                                                             numlma,
                                                                                                             nzm_lma,
                                                                                                             nzm_cowoutlet,
                                                                                                             dt,
                                                                                                             Tduration,
                                                                                                             Cval, Eval,
                                                                                                             Avalve,
                                                                                                             arterynode_map,
                                                                                                             i_upper_or_lower,
                                                                                                             artery_to_0dindex,
                                                                                                             sim_LMA,
                                                                                                             disregard_LMA)

        niter_0dcount += 1

        if niter_0dcount == iterrelax_0d:
            c_relax = c_relax / 5.0
        if niter_0dcount == iterrelax_0d * 3:
            c_relax = c_relax / 10.0

        Qguess[1:nartery + 1] = Qguess[1:nartery + 1] * 0.000001
        P1u[1:nartery + 1] = P1u[1:nartery + 1] * mmhgtoPa - p0

        C_ratto = 0.0
        Dif_ratio.fill(0.0)
        Q_temp.fill(0.0)
        A_temp.fill(0.0)

        C_ratto, Avalve = Crattocal(nartery, mbif_par, imaxtree, P1u, tbeta, A0, roi, Qguess, Dif_ratio,
                                    W1, W2, q, A_temp, Q_temp, C_ratto, Avalve, A_term1, remuda0)

        if C_ratto < converge_cri_coup:
            break

        else:
            Qguess[1:nartery + 1] = Qguess[1:nartery + 1] + (Q_temp[1:nartery + 1] - Qguess[1:nartery + 1]) * c_relax

    # replace end nodes
    if Atreem[1, 0] == 0:
        print("ERROR interface_01d: Atreem[1,0] is zero.")

    Atreem[1, 0] = A_temp[1]
    Qtreem[1, 0] = Q_temp[1]
    Utreem[1, 0] = Qtreem[1, 0] / Atreem[1, 0]

    for n in range(2, nartery + 1):
        if mbif_par[n, 0] in [0, 4]:
            Atreem[n, imaxtree[n]] = A_temp[n]
            Qtreem[n, imaxtree[n]] = Q_temp[n]

            if Atreem[n, imaxtree[n]] == 0:
                print("ERROR interface_01d: Atreem is zero for artery", n)

            Utreem[n, imaxtree[n]] = Qtreem[n, imaxtree[n]] / Atreem[n, imaxtree[n]]
        else:
            continue

    return (Atreem, Qtreem, Utreem, Vtree0d, Qtree0d, Qguess, P_0d, q, v, dv, dvq,
            Ptree0d, niter_0dcount, result, Cval, Eval, Avalve)


@njit(fastmath=True)
def Qguesscal(nartery, mbif_par, imaxtree, Atree, Qtree, A0, Qtreem, dt, roi, dx, Qguess,
              W1, W2, tbeta, A_term1, remuda0):
    for n in range(1, nartery + 1):
        if n == 1:
            j = 0
            Ax0 = Atree[n, 0]
            Ax00 = A0[n, 0]
            Ax1 = Atree[n, 1]
            Ax10 = A0[n, 1]

            # Debug checks for division by zero or sqrt of negative/zero
            if Ax0 <= 0 or Ax00 <= 0 or Ax1 <= 0 or Ax10 <= 0:
                print("ERROR Qguesscal n=1: Zero area detected. Ax0:", Ax0, "Ax00:", Ax00, "Ax1:", Ax1, "Ax10:", Ax10)
            if dx == 0:
                print("ERROR Qguesscal n=1: dx is zero.")
            if tbeta[n, j] == 0:
                print("ERROR Qguesscal n=1: tbeta is zero.")

            tbeta0 = tbeta[n, 0] * sqrt(Ax00)
            tsq0 = sqrt(tbeta0 * roi * 0.5 / Ax00)
            tbeta1 = tbeta[n, 1] * sqrt(Ax10)
            tsq1 = sqrt(tbeta1 * roi * 0.5 / Ax10)
            remd0 = Qtree[n, 0] / Ax0 + tsq0 * sqrt(sqrt(Ax0))
            remd1 = Qtree[n, 1] / Ax1 + tsq1 * sqrt(sqrt(Ax1))
            x_d_t = (remd0 + remd1) * 0.5 * dt
            rieman0 = Qtree[n, 0] / Ax0 - 4.0 * tsq0 * (sqrt(sqrt(Ax0)) - sqrt(sqrt(Ax00)))
            rieman1 = Qtree[n, 1] / Ax1 - 4.0 * tsq1 * (sqrt(sqrt(Ax1)) - sqrt(sqrt(Ax10)))

            W2[n] = (rieman0 - rieman1) * (dx - x_d_t) / dx + rieman1
            Qguess[n] = 2.0 * Qtreem[n, j + 1] - Qtreem[n, j + 2]

            A_term1[n] = (2.0 * sqrt(A0[n, j]) / tbeta[n, j] / roi) ** 2
            remuda0[n] = sqrt(tbeta[n, j] * roi * 0.5)

        elif mbif_par[n, 0] in [0, 4]:
            j = imaxtree[n]
            Ax0 = Atree[n, j - 1]
            Ax00 = A0[n, j - 1]
            Ax1 = Atree[n, j]
            Ax10 = A0[n, j]

            # Debug checks
            if Ax0 <= 0 or Ax00 <= 0 or Ax1 <= 0 or Ax10 <= 0:
                print("ERROR Qguesscal n=", n, ": Zero area. Ax0:", Ax0, "Ax00:", Ax00, "Ax1:", Ax1, "Ax10:", Ax10)
            if dx == 0:
                print("ERROR Qguesscal n=", n, ": dx is zero.")
            if tbeta[n, j] == 0:
                print("ERROR Qguesscal n=", n, ": tbeta is zero.")

            tbeta0 = tbeta[n, j - 1] * sqrt(Ax00)
            tsq0 = sqrt(tbeta0 * roi * 0.5 / Ax00)
            tbeta1 = tbeta[n, j] * sqrt(Ax10)
            tsq1 = sqrt(tbeta1 * roi * 0.5 / Ax10)
            remd0 = Qtree[n, j - 1] / Ax0 + tsq0 * sqrt(sqrt(Ax0))
            remd1 = Qtree[n, j] / Ax1 + tsq1 * sqrt(sqrt(Ax1))
            x_d_t = (remd0 + remd1) * 0.5 * dt
            rieman0 = Qtree[n, j - 1] / Ax0 + 4.0 * tsq0 * (sqrt(sqrt(Ax0)) - sqrt(sqrt(Ax00)))
            rieman1 = Qtree[n, j] / Ax1 + 4.0 * tsq1 * (sqrt(sqrt(Ax1)) - sqrt(sqrt(Ax10)))
            W1[n] = (rieman1 - rieman0) * (dx - x_d_t) / dx + rieman0
            Qguess[n] = 2.0 * Qtreem[n, j - 1] - Qtreem[n, j - 2]

            A_term1[n] = (2.0 * sqrt(A0[n, j]) / tbeta[n, j] / roi) ** 2
            remuda0[n] = sqrt(tbeta[n, j] * roi * 0.5)

        else:
            continue

    return Qguess, W1, W2, A_term1, remuda0


@njit(fastmath=True)
def Crattocal(nartery, mbif_par, imaxtree, P1u, tbeta, A0, roi, Qguess, Dif_ratio,
              W1, W2, q, A_temp, Q_temp, C_ratto, Avalve, A_term1, remuda0):
    for n in range(1, nartery + 1):

        if n == 1:
            j = 0
            if tbeta[n, j] == 0:
                print("ERROR Crattocal n=1: tbeta is zero.")

            A_temp_temp = P1u[n] / tbeta[n, j] + 1.0
            A_temp[n] = A0[n, j] * (A_temp_temp ** 2)

            if A_term1[n] <= 0:
                print("ERROR Crattocal n=1: A_term1 is zero or negative:", A_term1[n])

            A_t = sqrt(sqrt(A_temp[n] / A_term1[n]))
            W1[n] = W2[n] + (A_t - remuda0[n]) * 8.0
            Q_temp[n] = A_temp[n] * (W1[n] + W2[n]) * 0.5
            converge_c = Q_temp[n] - Qguess[n]
            if q[6] <= 0.0:
                Avalve[0] = 0.0

        elif mbif_par[n, 0] in [0, 4]:

            j = imaxtree[n]
            if tbeta[n, j] == 0:
                print("ERROR Crattocal n=", n, ": tbeta is zero.")

            A_temp_temp = P1u[n] / tbeta[n, j] + 1.0
            A_temp[n] = A0[n, j] * (A_temp_temp ** 2)

            if A_term1[n] <= 0:
                print("ERROR Crattocal n=", n, ": A_term1 is zero or negative:", A_term1[n])

            A_t = sqrt(sqrt(A_temp[n] / A_term1[n]))
            W2[n] = W1[n] - (A_t - remuda0[n]) * 8.0
            Q_temp[n] = A_temp[n] * (W1[n] + W2[n]) * 0.5
            converge_c = Q_temp[n] - Qguess[n]

        else:
            continue

        if abs(Qguess[n]) == 0.0:
            Dif_ratio[n] = 0.0
        else:
            # Check for safety even though the if-statement handles it
            if Qguess[n] == 0:
                print("ERROR Crattocal n=", n, ": Qguess is zero during Dif_ratio calculation.")
            Dif_ratio[n] = abs(converge_c) / abs(Qguess[n])

        C_ratto += Dif_ratio[n]

    return C_ratto, Avalve

# function for simulation loop !!!Modulerization Required!!!
# 0d calculation 
@njit(fastmath=True)
def zerodmodel(nnn,nartery,mbif_par,niter_0dcount,nzeromodel,Qguess,
               result,tee,tac,tar,tcr,q,v,dv,dvq,rukuk,
               P_0d,Ptree0d,Qtree0d,Vtree0d,P1u,RLCtree,numlma,nzm_lma,nzm_cowoutlet,
               dt,Tduration,Cval,Eval,Avalve,arterynode_map,i_upper_or_lower,artery_to_0dindex,sim_LMA,disregard_LMA):
    
    # Unit conversion from 1d to 0d
    Qguess[1:nartery+1] = Qguess[1:nartery+1] / 1.0e-6 # [ml/s] to [m^3/s]

    # Start computation

    # calculate and update values of nonlinear cardiac variables
    # compute the elastaces of pulmatory circulation
    Eval = update_eval(Eval, v)

    # compute calc_dv_dt to obtain current values of dv/dt
    dvq = calc_dv_dt(dvq, q, Qtree0d, Qguess, Eval)
    
    # update dv
    for i in range(0,20):
        dv[i] = dvq[2*i+1] # dV/dt

    # update nonlinear cardiac variables
    Cval = update_cat1(Cval, tcr, tee, tar, tac, Tduration, v, result)

    # implement 4th-order Runge-Kutta method
    result, Avalve = zd_rungekutta(rukuk,dvq,q,v,dv,Qtree0d,Qguess,Vtree0d,RLCtree,dt,result,
                              tcr,tee,Tduration,nzeromodel,nartery,numlma,nzm_lma,nzm_cowoutlet,Cval,Eval,Avalve,
                              arterynode_map,i_upper_or_lower,artery_to_0dindex,sim_LMA,disregard_LMA)

    # update q, v for next step calculation
    for i in range(0,20):
        q[i] = result[2*i,1] # Q
        v[i] = result[2*i+1,1] # V
    for i in range(40,nzm_lma,2):
        j = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
        Vtree0d[j] = result[i+1,1]
        Qtree0d[j] = result[i,1]
    if sim_LMA == 1: # if LMA is simulated
        for i in range(0,7):
            Qtree0d[nartery + i + 1] = result[i+112,1] # update Qtree0d for LMA
        for i in range(nzm_cowoutlet, nzeromodel, 2): # for cowoutlet nodes
            j = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
            Vtree0d[j] = result[i+1,1]
            Qtree0d[j] = result[i,1]

    # update pressure at the end of the 0D model for coupling calculation: P1u 
    P1u.fill(0.0) # reset P1u array
    # aortic inlet
    Eaa = Eval[0] # Eaa
    P1u[1] = Eaa * Zaa + Saa * dv[7] + pit
    # arterial end
    if sim_LMA == 0:
        for i in range(2,nartery+1):
            if mbif_par[i,0] in [0,4]:
                P1u[i] = Vtree0d[i] * RLCtree[3,i,1] + Qguess[i] * RLCtree[2,i,1]
    elif sim_LMA == 1:
        for i in range(6): # cow outlets
            arteryno = arterynode_map[40+2*i] # lookup_value(arterynode, 1, i, 0)
            arteryno2 = arterynode_map[119+2*i] # lookup_value(arterynode, 1, i, 0)
            P1u[arteryno] = Vtree0d[arteryno2] * RLCtree[3,arteryno,1] + Qguess[arteryno] * RLCtree[2,arteryno,1]
        for i in range(52,nzm_lma,2): # other terminal arteries
            arteryno = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
            if mbif_par[arteryno,0] in [0,4]:
                P1u[arteryno] = Vtree0d[arteryno] * RLCtree[3,arteryno,1] + Qguess[arteryno] * RLCtree[2,arteryno,1]
    # calculate the pressure P_0d and Ptree0d at the end of the 0D model for output
    P_0d, Ptree0d = p0dcal(P_0d,Ptree0d,v,dv,arterynode_map,result,RLCtree,Cval,Eval,sim_LMA,nzeromodel)

    return (result,P1u,q,v,dv,dvq,P_0d,Ptree0d,Qtree0d,Qguess,Vtree0d,Cval,Eval,Avalve)

@njit(fastmath=True)
def zd_rungekutta(rukuk,dvq,q,v,dv,Qtree0d,Qguess,Vtree0d,RLCtree,dt,result,tcr,tee,Tduration,
                  nzeromodel,nartery,numlma,nzm_lma,nzm_cowoutlet,Cval,Eval,Avalve,arterynode_map,
                  i_upper_or_lower,artery_to_0dindex,sim_LMA,disregard_LMA):

    for rank in range(0,4):   

        # calculate dv, dq using the government equation
        dvq = zerod_gov_eq(dvq,q,v,dv,Qtree0d,Qguess,Vtree0d,RLCtree,nartery,numlma,
                 tcr,Tduration,Cval,Eval,Avalve,arterynode_map,i_upper_or_lower,artery_to_0dindex,sim_LMA,disregard_LMA)
              
        rukuk[:,rank] = dvq[:] * dt # save the result(df/dt*dt) for each rank

        # update dv for next rank
        for i in range(0,20):
            dv[i] = dvq[2*i+1] # dV/dt

        # update q, v for next rank
        if rank in [0, 1]: # for next rank:2,3 OLD:rank<3
            for i in range(0,20):
                q[i] = result[2*i,0] + rukuk[2*i,rank] * 0.5 # Q
                v[i] = result[2*i+1,0] + rukuk[2*i+1,rank] * 0.5 # V
            for i in range(40,nzm_lma,2):
                j = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
                Vtree0d[j] = result[i+1,0] + rukuk[i+1,rank] * 0.5
                Qtree0d[j] = result[i,0] + rukuk[i,rank] * 0.5
            if sim_LMA == 1: # if LMA is simulated
                for i in range(0,7):
                    Qtree0d[nartery + 1 + i] = result[i+112,0] + rukuk[i+112,rank] * 0.5
                for i in range(nzm_cowoutlet, nzeromodel, 2): # for cowoutlet nodes
                    j = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
                    Vtree0d[j] = result[i+1,0] + rukuk[i+1,rank] * 0.5
                    Qtree0d[j] = result[i,0] + rukuk[i,rank] * 0.5

        elif rank == 2: # for next rank:4(last rank) OLD:rank==3
            for i in range(0,20):
                q[i] = result[2*i,0] + rukuk[2*i,rank] # Q
                v[i] = result[2*i+1,0] + rukuk[2*i+1,rank] # V
            for i in range(40,nzm_lma,2):
                j = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
                Vtree0d[j] = result[i+1,0] + rukuk[i+1,rank]
                Qtree0d[j] = result[i,0] + rukuk[i,rank]
            if sim_LMA == 1: # if LMA is simulated
                for i in range(0,7):
                    Qtree0d[nartery + 1 + i] = result[i+112,0] + rukuk[i+112,rank]      
                for i in range(nzm_cowoutlet, nzeromodel, 2): # for cowoutlet nodes
                    j = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
                    Vtree0d[j] = result[i+1,0] + rukuk[i+1,rank]
                    Qtree0d[j] = result[i,0] + rukuk[i,rank]

        # update the cardiac variables for the valves if rank = 0
        if rank == 0:
            Avalve = update_valve(Avalve, Eval, Cval, q, v, dv)

    # calculate the next state
    result[:,1] = result[:,0] + (rukuk[:,0] + 
            2.0 * (rukuk[:,1] + rukuk[:,2]) + rukuk[:,3]) / 6.0
        
    # set blood flow to 0 if the valve is closed
    # set valve variables
    aav = Avalve[0]
    amv = Avalve[1]
    atv = Avalve[2]
    apv = Avalve[3]
    verysmall = 1.0e-8
    verysmall2 = 2.0e-5
    if aav == 0.0 and result[12,0] <= verysmall: # left atrium
        result[12,1] = 0.0
    if (abs(tcr - Tduration) <= verysmall2 or tcr < 0.1 or
        (amv == 0.0 and result[10,0] <= verysmall)):
        result[10,1] = 0.0
    if apv == 0.0 and result[2,0] <= verysmall: # right atrium
        result[2,1] = 0.0
    if (abs(tcr - Tduration) <= verysmall2 or tcr < 0.1 or
        (atv == 0.0 and result[0,0] <= verysmall)):
        result[0,1] = 0.0

    return (result, Avalve)

@njit(fastmath=True)
def p0dcal(P_0d,Ptree0d,v,dv,arterynode_map,result,RLCtree,Cval,Eval,sim_LMA,nzeromodel):

    Eaa = Eval[0]
    Epua = Eval[1]
    Epuc = Eval[2]
    Epuv = Eval[3]
    Epwc = Eval[4]
    Epwv = Eval[5]
    ela = Cval[3]
    era = Cval[5]
    plv = Cval[6]
    prv = Cval[7]
    sla = Cval[8]
    slv = Cval[9]
    sra = Cval[10]
    srv = Cval[11]
    ppc = Cval[12]

    # Calculate P_0d (node number same as v): for output, only the ones not at the end of 0d model
    P_0d[0] = era * v[0] + sra * dv[0] + ppc + pit
    P_0d[1] = prv + srv * dv[1] + ppc + pit
    P_0d[2] = Epua * Zpua + Spua * dv[2] + pit
    P_0d[3] = Epuc * Zpuc + Spuc * dv[3] + pit
    P_0d[4] = Epuv * Zpuv + Spuv * dv[4] + pit
    P_0d[5] = ela * v[5] + sla * dv[5] + ppc + pit
    P_0d[6] = plv + slv * dv[6] + ppc + pit
    P_0d[7] = Eaa * Zaa + Saa * dv[7] + pit
    P_0d[8] = v[8] * invC_vcu + S_vcu * dv[8] + pit
    P_0d[9] = v[9] * invC_vcuu + S_vcuu * dv[9]
    P_0d[10] = v[10] * invC_upperv + S_upperv * dv[10]
    P_0d[11] = v[11] * invC_upperc + S_upperc * dv[11]
    P_0d[12] = v[12] * invC_upper + S_upper * dv[12]
    P_0d[13] = v[13] * invC_vcl + S_vcl * dv[13] + pit
    P_0d[14] = v[14] * invC_vclu + S_vclu * dv[14]
    P_0d[15] = v[15] * invC_lowerv + S_lowerv * dv[15]
    P_0d[16] = v[16] * invC_lowerc + S_lowerc * dv[16]
    P_0d[17] = v[17] * invC_lower + S_lower * dv[17]
    P_0d[18] = Epwc * Zpwc + Spwc * dv[18] + pit
    P_0d[19] = Epwv * Zpwv + Spwv * dv[19] + pit
    # calculate Ptree0d at the end of the 0D model for output
    for i in range(40, 112, 2):
        arteryno = arterynode_map[i] # lookup_value(arterynode, 1, i, 0)
        Ptree0d[arteryno] = result[i+1,1] * RLCtree[3,arteryno,1]
    if sim_LMA == 1: # if LMA is simulated, Ptree0d 91-96
        for i in range(6):
            idx1 = 119 + 2 * i 
            arteryno = arterynode_map[40 + 2 * i] # lookup_value(arterynode, 1, i, 0)
            idx2 = 91 + i
            Ptree0d[idx2] = result[idx1+1,1] * RLCtree[3,arteryno,1]
    
    return (P_0d, Ptree0d)

# Function to calculate the derivative of the state variables Q, V from the governing equations
@njit(fastmath=True)
def zerod_gov_eq(dvq,q,v,dv,Qtree0d,Qguess,Vtree0d,RLCtree,nartery,numlma,
                 tcr,Tduration,Cval,Eval,Avalve,arterynode_map,i_upper_or_lower,artery_to_0dindex,sim_LMA,disregard_LMA):

    # # run governing equations
    dvq = calc_dv_dt(dvq, q, Qtree0d, Qguess, Eval)
    dvq = calc_terminal_dv_dt(dvq, Qguess, Qtree0d, artery_to_0dindex, arterynode_map, sim_LMA, disregard_LMA)
    dvq = calc_dq_dt(dvq, q, v, dv, Eval, Cval)
    dvq = calc_terminal_dq_dt(dvq, v, Vtree0d, Qtree0d, numlma, RLCtree, arterynode_map, i_upper_or_lower,sim_LMA,disregard_LMA)
    
    # set blood flow to 0 if the valve is closed 
    aav = Avalve[0]
    amv = Avalve[1]
    atv = Avalve[2]
    apv = Avalve[3]
    verysmall = 1.0e-8
    verysmall2 = 2.0e-5
    if aav == 0.0 and q[6] <= verysmall: # left atrium
        dvq[12] = 0.0
    if (abs(tcr - Tduration) <= verysmall2 or tcr < 0.1 or 
        (amv == 0.0 and q[5] <= verysmall)):
        dvq[10] = 0.0
    if apv == 0.0 and q[1] <= verysmall: # right atrium
        dvq[2] = 0.0
    if (abs(tcr - Tduration) <= verysmall2 or tcr < 0.1 or
        (atv == 0.0 and q[0] <= verysmall)):
        dvq[0] = 0.0

    # return dq/dt and dv/dt
    return dvq

@njit(fastmath=True)
def calc_dv_dt(dvq, q, Qtree0d, Qguess, Eval):

    Eaa = Eval[0]  # Eaa

    # calculate dv/dt (2k+1)
    dvq[3] = q[0] - q[1]
    dvq[5] = q[1] - q[2] - q[17]
    dvq[7] = q[2] - q[3]
    dvq[9] = q[3] - q[4]
    dvq[11] = q[4] + q[19] - q[5]
    dvq[13] = q[5] - q[6]
    dvq[15] = q[6] - Qguess[1]  # qco=0
    dvq[1] = q[7] + q[12] - q[0]  # qco=0
    dvq[17] = q[8] - q[7]
    dvq[19] = q[9] - q[8]
    dvq[21] = q[10] - q[9]
    dvq[23] = q[11] - q[10]
    # dvq[25] = (Qtree0d[58] + Qtree0d[61] + Qtree0d[63] + Qtree0d[65] +
    #            Qtree0d[67] + Qtree0d[70] + Qtree0d[74] + Qtree0d[75] +
    #            Qtree0d[78] + Qtree0d[79] + Qtree0d[80] + Qtree0d[81] +
    #            Qtree0d[82] + Qtree0d[83] + Qtree0d[8] + Qtree0d[19] +
    #            Qtree0d[43] + Qtree0d[44] + Qtree0d[45] + Qtree0d[46] - q[11])

    dvq[25] = -q[11]
    dvq[35] = -q[16]
    for i in range(len(arterynode[:,0])):
        if arterynode[i,2] in [0,2]:  # if upper body or cerebral
            dvq[25] += Qtree0d[arterynode[i,0]]
        elif arterynode[i,2] == 1:  # if lower body
            dvq[35] += Qtree0d[arterynode[i,0]]

    dvq[27] = q[13] - q[12]
    dvq[29] = q[14] - q[13]
    dvq[31] = q[15] - q[14]
    dvq[33] = q[16] - q[15]
    # dvq[35] = (Qtree0d[14] + Qtree0d[26] + Qtree0d[30] + Qtree0d[28] +
    #            Qtree0d[32] + Qtree0d[22] + Qtree0d[23] + Qtree0d[24] +
    #            Qtree0d[36] + Qtree0d[51] + Qtree0d[37] + Qtree0d[52] +
    #            Qtree0d[41] + Qtree0d[42] + Qtree0d[54] + Qtree0d[55] - q[16])
    dvq[37] = q[17] - q[18]
    dvq[39] = q[18] - q[19]

    return dvq

@njit(fastmath=True)
def calc_terminal_dv_dt(dvq, Qguess, Qtree0d, artery_to_0dindex, arterynode_map, sim_LMA, disregard_LMA):

    # if sim_LMA == 1:

    #     dvq[41] = Qguess[58] - Qtree0d[58] + Qtree0d[85] - Qtree0d[84]
    #     dvq[43] = Qguess[61] - Qtree0d[61] + Qtree0d[86] - Qtree0d[85]
    #     dvq[45] = Qguess[63] - Qtree0d[63] + Qtree0d[87] - Qtree0d[86] + Qtree0d[84]
    #     dvq[47] = Qguess[65] - Qtree0d[65] + Qtree0d[88] - Qtree0d[87] - Qtree0d[90]
    #     dvq[49] = Qguess[67] - Qtree0d[67] + Qtree0d[89] - Qtree0d[88]
    #     dvq[51] = Qguess[70] - Qtree0d[70] + Qtree0d[90] - Qtree0d[89]

    # elif sim_LMA == 0:
    #     dvq[41] = Qguess[58] - Qtree0d[58]
    #     dvq[43] = Qguess[61] - Qtree0d[61]
    #     dvq[45] = Qguess[63] - Qtree0d[63]
    #     dvq[47] = Qguess[65] - Qtree0d[65]
    #     dvq[49] = Qguess[67] - Qtree0d[67]
    #     dvq[51] = Qguess[70] - Qtree0d[70]

    if sim_LMA == 0:
        for i in range(40, 112, 2):
            arteryno = arterynode_map[i] # int(lookup_value(arterynode, 1, i, 0))     
            dvq[i+1] = Qguess[arteryno] - Qtree0d[arteryno] # Q4DFlow - Qspect

    elif sim_LMA == 1:

        for i in range(6): # dvq[41], dvq[43], dvq[45], dvq[47], dvq[49], dvq[51]
            idx = 40 + i * 2 
            arteryno = arterynode_map[idx] # int(lookup_value(arterynode, 1, i, 0))
            dvq[idx+1] = Qtree0d[91+i] - Qtree0d[arteryno]

        for i in range(len(LMA_connection)): # adjust dvq[41-51(odd)] for LMA connections
            LMAno = i + 84
            if disregard_LMA[i] == 1:
                continue
            arteryupper = LMA_connection[i,0]
            arterylower = LMA_connection[i,1]
            dvqupper = artery_to_0dindex[arteryupper] + 1
            dvqlower = artery_to_0dindex[arterylower] + 1
            dvq[dvqupper] -= Qtree0d[LMAno] # add or subtract LMA flow from connected arteries
            dvq[dvqlower] += Qtree0d[LMAno]

        for i in range(52,112,2): # dvq[53], dvq[55], ..., dvq[111]
            arteryno = arterynode_map[i] # int(lookup_value(arterynode, 1, i, 0))     
            dvq[i+1] = Qguess[arteryno] - Qtree0d[arteryno] # Q4DFlow - Qspect
        
        for i in range(6): # dvq[120], dvq[122], dvq[124], dvq[126], dvq[128], dvq[130]
            idx_up = 40 + i * 2
            idx = 119 + 2*i + 1
            arteryno_up = arterynode_map[idx_up] # int(lookup_value(arterynode, 1, i, 0))
            dvq[idx] = Qguess[arteryno_up] - Qtree0d[91+i]

    return dvq

@njit(fastmath=True)
def calc_dq_dt(dvq, q, v, dv, Eval, Cval):

    era = Cval[5]
    ela = Cval[3]
    ppc = Cval[12]
    prv = Cval[7]
    plv = Cval[6]
    Sra = Cval[10]
    Srv = Cval[11]
    Sla = Cval[8]
    Slv = Cval[9]
    Eaa = Eval[0]
    Epuc = Eval[2]
    Epua = Eval[1]
    Epuv = Eval[3]
    Epwc = Eval[4]
    Epwv = Eval[5]

    # Calculate dq/dt (2k)
    dvq[0] = (era * v[0] - prv - Rtv * q[0] - btv * q[0] * abs(q[0]) + Sra * dv[0] - Srv * dv[1]) * invytv
    dvq[2] = (prv - Epua * Zpua - Rpv * q[1] - bpv * q[1] * abs(q[1])
              + Srv * dv[1] - Spua * dv[2] + ppc) * invypv
    dvq[4] = (Epua * Zpua - Epuc * Zpuc - Rpua * q[2]
              + Spua * dv[2] - Spuc * dv[3]) * invypua
    dvq[6] = (Epuc * Zpuc - Epuv * Zpuv - Rpuc * q[3]
              + Spuc * dv[3] - Spuv * dv[4]) * invypuc
    dvq[8] = (Epuv * Zpuv - ela * v[5] - Rpuv * q[4]
              + Spuv * dv[4] - Sla * dv[5] - ppc) * invypuv
    dvq[10] = (ela * v[5] - plv - Rmv * q[5] - bmv * q[5] * abs(q[5])
               + Sla * dv[5] - Slv * dv[6]) * invymv
    dvq[12] = (plv - Eaa * Zaa - Rav * q[6] - bav * q[6] * abs(q[6])
               + Slv * dv[6] - Saa * dv[7] + ppc) * invyav
    dvq[14] = (v[8] * invC_vcu - era * v[0] - R_vcu1 * q[7] + S_vcu * dv[8]
               - Sra * dv[0] - ppc) * invy_vcu1
    dvq[16] = (v[9]*invC_vcuu - R_vcu * q[8] - v[8] * invC_vcu + S_vcuu * dv[9]
               - S_vcu * dv[8] - pit) * invy_vcu
    dvq[18] = (v[10] * invC_upperv - R_vcuu * q[9] - v[9] * invC_vcuu + S_upperv * dv[10]
               - S_vcuu * dv[9]) * invy_vcuu
    dvq[20] = (v[11] * invC_upperc - R_upperv * q[10] - v[10] * invC_upperv + S_upperc * dv[11]
               - S_upperv * dv[10]) * invy_upperv
    dvq[22] = (v[12] * invC_upper - R_upperc * q[11] - v[11] * invC_upperc + S_upper * dv[12]
               - S_upperc * dv[11]) * invy_upperc
    dvq[24] = (v[13] * invC_vcl - era * v[0] - R_vcl1 * q[12] + S_vcl * dv[13]
               - Sra * dv[0] - ppc) * invy_vcl1
    dvq[26] = (v[14] * invC_vclu - R_vcl * q[13] - v[13] * invC_vcl + S_vclu * dv[14]
               - S_vcl * dv[13] - pit) * invy_vcl
    dvq[28] = (v[15] * invC_lowerv - R_vclu * q[14] - v[14] * invC_vclu + S_lowerv * dv[15]
               - S_vclu * dv[14]) * invy_vclu
    dvq[30] = (v[16] * invC_lowerc - R_lowerv * q[15] - v[15] * invC_lowerv + S_lowerc * dv[16]
               - S_lowerv * dv[15]) * invy_lowerv
    dvq[32] = (v[17] * invC_lower - R_lowerc * q[16] - v[16] * invC_lowerc + S_lower * dv[17]
               - S_lowerc * dv[16]) * invy_lowerc
    if gpw > 0:
        dvq[34] = (Epua * Zpua - Epwc * Zpwc - q[17] / gpw
                   + Spua * dv[2] - Spwc * dv[18]) * invypwa
    else:
        dvq[34] = 0.0
    dvq[36] = (Epwc * Zpwc - Epwv * Zpwv - Rpwc * q[18] + Spwc * dv[18]
               - Spwv * dv[19]) * invypwc
    dvq[38] = (Epwv * Zpwv - ela * v[5] - Rpwv * q[19] + Spwv * dv[19]
               - Sla * dv[5] - ppc) * invypwv
    
    return dvq

@njit(fastmath=True)
def calc_terminal_dq_dt(dvq, v, Vtree0d, Qtree0d, numlma, RLCtree, arterynode_map, i_upper_or_lower, sim_LMA, disregard_LMA):

    if sim_LMA == 0:
        for i in range(40, 112, 2):
            arteryno = arterynode_map[i] # int(lookup_value(arterynode, 1, i, 0))
            discriminant = i_upper_or_lower[i] # int(lookup_value(arterynode, 1, i, 2))
            if discriminant == 0 or discriminant == 2:
                dvq[i] = (Vtree0d[arteryno] * RLCtree[3,arteryno,1] -
                        Qtree0d[arteryno] * RLCtree[2,arteryno,2] -
                        v[12] * invC_upper) * RLCtree[1,arteryno,2]
            elif discriminant == 1:
                dvq[i] = (Vtree0d[arteryno] * RLCtree[3,arteryno,1] -
                        Qtree0d[arteryno] * RLCtree[2,arteryno,2] -
                        v[17] * invC_lower) * RLCtree[1,arteryno,2]
    elif sim_LMA == 1:

        # dvq[112] = (Vtree0d[58]/RLCtree[3,58,1]-Vtree0d[63]/RLCtree[3,63,1]-RLCtree[2,84,1]*Qtree0d[84])/RLCtree[1,84,2] #qlma[0] caution, C/L is inverted here
        # dvq[113] = (Vtree0d[61]/RLCtree[3,61,1]-Vtree0d[58]/RLCtree[3,58,1]-RLCtree[2,85,1]*Qtree0d[85])/RLCtree[1,85,2] #qlma[1]
        # dvq[114] = (Vtree0d[63]/RLCtree[3,63,1]-Vtree0d[61]/RLCtree[3,61,1]-RLCtree[2,86,1]*Qtree0d[86])/RLCtree[1,86,2] #qlma[2]
        # dvq[115] = (Vtree0d[65]/RLCtree[3,65,1]-Vtree0d[63]/RLCtree[3,63,1]-RLCtree[2,87,1]*Qtree0d[87])/RLCtree[1,87,2] #qlma[3]
        # dvq[116] = (Vtree0d[67]/RLCtree[3,67,1]-Vtree0d[65]/RLCtree[3,65,1]-RLCtree[2,88,1]*Qtree0d[88])/RLCtree[1,88,2] #qlma[4]
        # dvq[117] = (Vtree0d[70]/RLCtree[3,70,1]-Vtree0d[67]/RLCtree[3,67,1]-RLCtree[2,89,1]*Qtree0d[89])/RLCtree[1,89,2] #qlma[5]
        # dvq[118] = (Vtree0d[65]/RLCtree[3,65,1]-Vtree0d[70]/RLCtree[3,70,1]-RLCtree[2,90,1]*Qtree0d[90])/RLCtree[1,90,2] #qlma[6]

        for i in range(numlma): # dvq[112] to dvq[118]: qlma[i]
            dvqindex = 112 + i
            arteryupper = LMA_connection[i,0]
            arterylower = LMA_connection[i,1]

            if disregard_LMA[i] == 0:
                dvq[dvqindex] = (Vtree0d[arteryupper]*RLCtree[3,arteryupper,1]-Vtree0d[arterylower]*RLCtree[3,arterylower,1]
                                -RLCtree[2,84+i,1]*Qtree0d[84+i])*RLCtree[1,84+i,2] 
            else:
                dvq[dvqindex] = 0.0
            
        for i in range(6): # dvq[119], dvq[121], dvq[123], dvq[125], dvq[127], dvq[129], dvq[40], dvq[42], dvq[44], dvq[46], dvq[48], dvq[50]
            dvqidx1 = 119 + 2 * i
            dvqidx2 = 40 + i * 2
            arterylower = arterynode_map[dvqidx2] # int(lookup_value(arterynode, 1, i, 0))
            arteryupper = 91 + i
            dvq[dvqidx1] = (Vtree0d[arteryupper]*RLCtree[3,arterylower,1] - Vtree0d[arterylower]*RLCtree[3,arterylower,1] # dvq[119], dvq[121], dvq[123], dvq[125], dvq[127], dvq[129]
                           - RLCtree[2,arterylower,2]*Qtree0d[arteryupper]) * RLCtree[1,arterylower,2]
            dvq[dvqidx2] = (Vtree0d[arterylower] * RLCtree[3,arterylower,1] - v[12] * invC_upper # dvq[40], dvq[42], dvq[44], dvq[46], dvq[48], dvq[50]
                      - RLCtree[2,arterylower,3]*Qtree0d[arterylower]) * RLCtree[1,arterylower,2]
        
        for i in range(52,112,2): # dvq[52] to dvq[110](even)
            arteryno = arterynode_map[i] # int(lookup_value(arterynode, 1, i, 0))
            discriminant = i_upper_or_lower[i] # int(lookup_value(arterynode, 1, i, 2))
            if discriminant == 0 or discriminant == 2:
                dvq[i] = (Vtree0d[arteryno] * RLCtree[3,arteryno,1] -
                        Qtree0d[arteryno] * RLCtree[2,arteryno,2] -
                        v[12] * invC_upper) * RLCtree[1,arteryno,2]
            elif discriminant == 1:
                dvq[i] = (Vtree0d[arteryno] * RLCtree[3,arteryno,1] -
                        Qtree0d[arteryno] * RLCtree[2,arteryno,2] -
                        v[17] * invC_lower) * RLCtree[1,arteryno,2]
    return dvq


