import numpy as np
from numba import njit

exp = np.exp
sqrt = np.sqrt

def tbetacal(tbeta, cof_ela, Rtree0, imaxtree, nartery, age, roi, cow_geo, exclude_artery, c_ela_aor):

    # parameter setting
    Ek1 = 2000000.0 # [kg/(s^2*m)]
    Ek2 = -2253.0 # [m-1]
    Ek3 = 86500.0 # [kg/(s^2*m)]
    c_ela = 4.0 / 3.0 # 1/(1-ξ^2) (ξ=1/2)

    # calculate cof_ela for each artery

    # for aorta trunk
    r_bas=0.01525 # literatude value of aorta radius [m]
    e_modify = (Ek1 * exp(Ek2 * r_bas) + Ek3) / (Ek1 * exp(Ek2 * Rtree0[1,0]) + Ek3) # (aorta literature eh/r)/(aorta patient specific eh/r)
    c_ela_aor = c_ela_aor * e_modify # elastance of aorta
    c_aging = 0.48 * c_ela_aor # coefficient of aorta trunk cosiedering aging
    for i in [1,2,10,12,13,25,27,29,31,33]: # aorta trunk
        cof_ela[i] = c_ela * c_aging

    # Splancnic and renal circulation
    c_ela_aging = 1.07 * (c_ela_aor ** 1.0)
    cof_ela[14] = c_ela * c_ela_aging * 0.85
    cof_ela[20] = c_ela * c_ela_aging * 0.58
    cof_ela[21] = c_ela * c_ela_aging * 0.70
    cof_ela[22] = c_ela * c_ela_aging * 0.76
    cof_ela[23] = c_ela * c_ela_aging * 0.56
    cof_ela[24] = c_ela * c_ela_aging * 0.56
    cof_ela[26] = c_ela * c_ela_aging * 0.57
    cof_ela[28] = c_ela * c_ela_aging * 0.6
    cof_ela[30] = c_ela * c_ela_aging * 0.6
    cof_ela[32] = c_ela * c_ela_aging * 0.5

    # for Upper limb
    for j in [7,8,9]:
        i = int(imaxtree[j] / 2)
        if j == 7:
            cof_ela[j] = c_ela * 1.3 * 0.4
        elif j == 8:
            cof_ela[j] = c_ela * 1.4 * 0.8
        elif j == 9:
            cof_ela[j] = c_ela * 2.0 * 0.6
        tbeta[j,i] = (Ek1 * exp(Ek2 * Rtree0[j,i]) + Ek3) * cof_ela[j] # tbeta(i,j):Eh/(r0(1-ξ^2))
        v_wave0 = sqrt(tbeta[j,i] * roi / 2.0) # wave speed c = sqrt(Eh/(2ρr0(1-ξ^2)))
        v_wave = (age - 25.0) * 0.038 + v_wave0 # wave speed of each artery considering the age [m/s]
        if j == 7:
            c_ela_aging_ul = (v_wave / v_wave0) ** 2.0 # aging coefficient of wave speed
        elif j == 8:
            c_res_aging_ul1 = (v_wave / v_wave0) ** 2.0 # aging coefficient of wave speed
        elif j == 9:
            c_res_aging_ul2 = (v_wave / v_wave0) ** 2.0 # aging coefficient of wave speed
    for j in [4,15,6,16,7,17,8,19,9,18,43,46,44,45]: # upper limb circulation 
        if j in [4,15]:
            cof_ela[j] = c_ela * c_ela_aging * 0.6 * 0.8 #bug
        elif j in [6,16]:
            cof_ela[j] = c_ela * c_res_aging_ul1 * 1.7 * 0.8 #bug
        elif j in [7,17]:
            cof_ela[j] = c_ela * c_ela_aging_ul * 1.3 * 0.7 #bug
        elif j in [8,19]:
            cof_ela[j] = c_ela * c_res_aging_ul1 * 1.4 * 0.8 #bug
        elif j in [9,18]:
            cof_ela[j] = c_ela * c_res_aging_ul2 * 2.0 * 0.6 #bug
        elif j in [43,46]:
            cof_ela[j] = c_ela * 1.3 #bug
        elif j in [44,45]:
            cof_ela[j] = c_ela * c_res_aging_ul1 * 2.2 #bug

    # for Lower limb
    c_ll = 0.65
    for j in [38,41,42]:
        i = int(imaxtree[j] / 2)
        if j == 38:
            cof_ela[j] = c_ela * c_ll * 1.8
        elif j == 41:
            cof_ela[j] = c_ela * c_ll * 2.8
        elif j == 42:
            cof_ela[j] = c_ela * c_ll * 2.5
        tbeta[j,i] = (Ek1 * exp(Ek2 * Rtree0[j,i]) + Ek3) * cof_ela[j] # tbeta(i,j):Eh/(r0(1-ξ^2))
        v_wave0 = sqrt(tbeta[j,i] * roi / 2.0) # wave speed c = sqrt(Eh/(2ρr0(1-ξ^2)))
        v_wave = (age - 25.0) * 0.045 + v_wave0 # wave speed of each artery considering the age [m/s]
        if j == 38:
            c_ela_aging_ll = (v_wave / v_wave0) ** 2.0 # aging coefficient of wave speed
        elif j == 41:
            c_res_aging_ll1 = (v_wave / v_wave0) ** 2.0 # aging coefficient of wave speed
        elif j == 42:
            c_res_aging_ll2 = (v_wave / v_wave0) ** 2.0 # aging coefficient of wave speed
    for j in [34,49,36,51,35,50,37,52,38,53,41,54,42,55]: # lower limb circulation
        if j in [34,49]:
            cof_ela[j] = c_ela * c_ll * c_ela_aging  * 0.53 * 1.5 
        elif j in [35,50]:
            cof_ela[j] = c_ela * c_ll * c_ela_aging * 1.3 * 1.0 
        elif j in [36,51]:
            cof_ela[j] = c_ela * c_ll * c_ela_aging_ll * 2.3 
        elif j in [37,52]:
            cof_ela[j] = c_ela * c_ll * c_ela_aging_ll * 1.4 
        elif j in [38,53]:
            cof_ela[j] = c_ela * c_ll * c_ela_aging_ll * 1.8 
        elif j in [41,54]:
            cof_ela[j] = c_ela * c_ll * c_res_aging_ll1 * 2.8 
        elif j in [42,55]:
            cof_ela[j] = c_ela * c_ll * c_res_aging_ll2 * 2.5 

    # for cerebral circulation
    for i in [3,5,11]: # cerebral circulation 1
        cof_ela[i] = c_ela * c_ela_aging * 0.6
    for i in [39,48,47,40]: # cerebral circulation 2
        cof_ela[i] = c_ela * c_ela_aging_ll * 1.0 

    # debug
    # for j in range(1,nartery+1):
    #     print(f"cof_ela[{j}] = {cof_ela[j]}")

    # calculate tbeta(i,j):Eh/(r0(1-ξ^2)) of each artery and each node
    # adjust cof_ela[j] if wave speed > 35.0 m/s
    for j in range(1,nartery+1):
        i = int(imaxtree[j] / 2)
        tbeta[j,i] = (Ek1 * exp(Ek2 * Rtree0[j,i]) + Ek3) * cof_ela[j] # Eh/(r0(1-ξ^2))
        v_wave = sqrt(tbeta[j,i] * roi / 2.0) # wave speed c = sqrt(Eh/(2ρr0(1-ξ^2)))
        if (v_wave >= 35.0): #adjust if PWV > 35.0
            cof_ela[j] = cof_ela[j] * ((35.0 / v_wave) ** 2.0) 
    # calculate for each node
    for j in range(1,nartery+1):
        if (cow_geo != 0 and exclude_artery[j] == 1):
            continue
        for i in range(0, imaxtree[j]+1):
            tbeta[j,i] = (Ek1 * exp(Ek2 * Rtree0[j,i]) + Ek3) * cof_ela[j] # Eh/(r0(1-ξ^2))
            if (i == int(imaxtree[j] / 2)):
                v_wave = sqrt(tbeta[j,i] * roi / 2.0) # wave speed c = sqrt(Eh/(2ρr0(1-ξ^2)))
                print(f"{j}th artery Eh/(r(1-ξ^2)) = {tbeta[j,i]}, wave speed c = {v_wave}")

    return tbeta, cof_ela, c_ela_aor