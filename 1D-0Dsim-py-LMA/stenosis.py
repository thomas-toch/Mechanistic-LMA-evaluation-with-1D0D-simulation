import numpy as np
import math
from numba import njit

exp = math.exp
sqrt = math.sqrt
pow = math.pow

# Stenosis computation : regard stenosis as a 0D model in between 1D models
@njit(fastmath=True)
def interface_stn(nnn,numb_stn,nst_ran,byr_st,Atreem,Qtreem,Utreem,Ptreem,Atree,Qtree,Utree,A0,p0,
                  dx,dt,roi,tbeta,c_relax,converge_cri,itermax_stn):

    for nn in range(1,numb_stn+1): # loop for each stenosed artery
    
        n = nst_ran[nn,0] # artery no.
        jstart = nst_ran[nn,1] # start node no.
        jend = nst_ran[nn,2] # end node no.
        stn_len = float(jend - jstart) * dx
        jneck = nst_ran[nn,3] # neck node no.

        # calculate the reimann variables for the start and the end node of the stenosis
        j = jstart
        Ax0 = Atree[n,j-1]
        Ax00 = A0[n,j-1]
        Ax1 = Atree[n,j]
        Ax10 = A0[n,j]
        # tbeta0 = (Ek1 * exp(Ek2 * Rtree0[j-1,n]) + Ek3) * sqrt(Ax00) * cof_ela[n]
        tbeta0 = tbeta[n,j-1] * sqrt(Ax00)
        tsq0 = sqrt(tbeta0 * roi * 0.5 / Ax00)
        # tbeta1 = (Ek1 * exp(Ek2 * Rtree0[j,n]) + Ek3) * sqrt(Ax10) * cof_ela[n]
        tbeta1 = tbeta[n,j] * sqrt(Ax10)
        tsq1 = sqrt(tbeta1 * roi * 0.5 / Ax10)
        remd0 = Qtree[n,j-1] / Ax0 + tsq0 * sqrt(sqrt(Ax0))
        remd1 = Qtree[n,j] / Ax1 + tsq1 * sqrt(sqrt(Ax1))
        x_d_t = (remd0 + remd1) * 0.5 * dt
        rieman0 = Qtree[n,j-1] / Ax0 + 4.0 * tsq0 * (sqrt(sqrt(Ax0)) - sqrt(sqrt(Ax00)))
        rieman1 = Qtree[n,j] / Ax1 + 4.0 * tsq1 * (sqrt(sqrt(Ax1)) - sqrt(sqrt(Ax10)))
        W1_stn = (rieman1 - rieman0) * (dx - x_d_t) / dx + rieman0
        Qguess1_st = 2.0 * Qtreem[n,j-1] - Qtreem[n,j-2] # Qguess1 = 2*Q(i-1) - Q(i-2)
        
        A_term_jstart = (2.0 * sqrt(A0[n,j]) / tbeta[n,j] / roi) ** 2
        remuda0_jstart = sqrt(tbeta[n,j] * roi * 0.5)

        j = jend
        Ax0 = Atree[n,j]
        Ax00 = A0[n,j]
        Ax1 = Atree[n,j+1]
        Ax10 = A0[n,j+1]
        # tbeta0 = (Ek1 * exp(Ek2 * Rtree0[j,n]) + Ek3) * sqrt(Ax00) * cof_ela[n]
        tbeta0 = tbeta[n,j] * sqrt(Ax00)
        tsq0 = sqrt(tbeta0 * roi * 0.5 / Ax00)
        # tbeta1 = (Ek1 * exp(Ek2 * Rtree0[j + 1,n]) + Ek3) * sqrt(Ax10) * cof_ela[n]
        tbeta1 = tbeta[n,j+1] * sqrt(Ax10)
        tsq1 = sqrt(tbeta1 * roi * 0.5 / Ax10)
        remd0 = Qtree[n,j] / Ax0 + tsq0 * sqrt(sqrt(Ax0))
        remd1 = Qtree[n,j+1] / Ax1 + tsq1 * sqrt(sqrt(Ax1))
        x_d_t = (remd0 + remd1) * 0.5 * dt
        rieman0 = Qtree[n,j] / Ax0 - 4.0 * tsq0 * (sqrt(sqrt(Ax0)) - sqrt(sqrt(Ax00)))
        rieman1 = Qtree[n,j+1] / Ax1 - 4.0 * tsq1 * (sqrt(sqrt(Ax1)) - sqrt(sqrt(Ax10)))
        W2_stn = (rieman0 - rieman1) * (dx - x_d_t) / dx + rieman1
        Qguess2_st = 2.0 * Qtreem[n,j+1] - Qtreem[n,j+2] # Qguess2 = 2*Q(i+1) - Q(i+2)

        A_term_jend = (2.0 * sqrt(A0[n,j]) / tbeta[n,j] / roi) ** 2
        remuda0_jend = sqrt(tbeta[n,j] * roi * 0.5)

        Qguess1_st = 0.4 * Qguess1_st + 0.6 * Qguess2_st 

        iter_stn = 0

        damp = 1.0 # damping factor for convergence
        nrmax = 6 # maximum number of iterations for newton's method (default:12)
        A1_base = Atree[n,jstart]
        A2_base = Atree[n,jend]
        A1_new = A1_base # initial value for A1
        A2_new = A2_base # initial value for A2
        A1_0 = A0[n,jstart]
        A2_0 = A0[n,jend]
        tbeta_jstart = tbeta[n,jstart]
        tbeta_jend = tbeta[n,jend]
        Q2_base = Qtree[n,jend]

        while iter_stn < itermax_stn: # repeat until converged
            
            iter_stn += 1

            Dif_ratio = 0.0 # initial value for convergence test
            
            # update A with Newton's method
            # jstart
            A1_new = A1_base # initial value for A1
            for i in range(nrmax):

                # f_a = A1_new - A_term_jstart * ((-Qguess1_st / A1_new / 4.0 + W1_stn / 4.0 + remuda0_jstart) ** 4)
                # f1_a = 1.0 - A_term_jstart * ((-Qguess1_st / A1_new / 4.0 + W1_stn / 4.0 + remuda0_jstart) ** 3) * Qguess1_st / (A1_new * A1_new)
                inv_A1_new = 1.0 / A1_new
                k1 = -Qguess1_st * inv_A1_new * 0.25 + W1_stn * 0.25 + remuda0_jstart
                k2 = k1 * k1
                k3 = k2 * k1
                k4 = k2 * k2
                f_a = A1_new - A_term_jstart * k4
                f1_a = 1.0 - A_term_jstart * k3 * Qguess1_st * inv_A1_new * inv_A1_new

                A1_new -= damp * f_a / f1_a
            P1_new = tbeta_jstart * (sqrt(A1_new / A1_0) - 1.0) + p0
            # jend
            A2_new = A2_base # initial value for A2
            for i in range(nrmax):

                # f_a = A2_new - A_term_jend * ((Qguess1_st / A2_new / 4.0 - W2_stn / 4.0 + remuda0_jend) ** 4)
                # f1_a = 1.0 + A_term_jend * ((Qguess1_st / A2_new / 4.0 - W2_stn / 4.0 + remuda0_jend) ** 3) * Qguess1_st / (A2_new * A2_new) 
                inv_A2_new = 1.0 / A2_new
                k1 = Qguess1_st * inv_A2_new * 0.25 - W2_stn * 0.25 + remuda0_jend
                k2 = k1 * k1
                k3 = k2 * k1
                k4 = k2 * k2
                f_a = A2_new - A_term_jend * k4
                f1_a = 1.0 + A_term_jend * k3 * Qguess1_st * inv_A2_new * inv_A2_new

                A2_new -= damp * f_a / f1_a
            P2_new = tbeta_jend * (sqrt(A2_new / A2_0) - 1.0) + p0

            # Compute new Q with Runge-Kutta method
            Q_new = Q2_base

            # for i in range(0,4):
            #     dp_st = r_st * Q_new + b_st * Q_new * abs(Q_new)
            #     dQ_stn[i] = (P1_new - dp_st - P2_new) / y_st
            #     if i == 0:
            #         Q_new = Q2_base + dt * 0.5 * dQ_stn[i]
            #     if i == 1:
            #         Q_new = Q2_base + dt * 0.5 * dQ_stn[i]
            #     if i == 2:
            #         Q_new = Q2_base + dt * dQ_stn[i]
            #     if i == 3:
            #         Q_new = Q2_base + dt / 6.0 * (dQ_stn[0] + 2.0 * dQ_stn[1] + 2.0 * dQ_stn[2] + dQ_stn[3])
            
            # dp_st = r_st * Q_new + b_st * Q_new * abs(Q_new)
            # dQ_1 = (P1_new - dp_st - P2_new) / y_st
            # Q_new = Q2_base + dt * 0.5 * dQ_1
            # dp_st = r_st * Q_new + b_st * Q_new * abs(Q_new)
            # dQ_2 = (P1_new - dp_st - P2_new) / y_st
            # Q_new = Q2_base + dt * 0.5 * dQ_2
            # dp_st = r_st * Q_new + b_st * Q_new * abs(Q_new)
            # dQ_3 = (P1_new - dp_st - P2_new) / y_st
            # Q_new = Q2_base + dt * dQ_3
            # dp_st = r_st * Q_new + b_st * Q_new * abs(Q_new)
            # dQ_4 = (P1_new - dp_st - P2_new) / y_st
            # Q_new = Q2_base + dt / 6.0 * (dQ_1 + 2.0 * dQ_2 + 2.0 * dQ_3 + dQ_4)

            dp_st = byr_st[nn,2] * Q_new + byr_st[nn,0] * Q_new * abs(Q_new)
            dQ_1 = (P1_new - dp_st - P2_new) * byr_st[nn,1]
            Q_new = Q2_base + dt * 0.5 * dQ_1
            dp_st = byr_st[nn,2] * Q_new + byr_st[nn,0] * Q_new * abs(Q_new)
            dQ_2 = (P1_new - dp_st - P2_new) * byr_st[nn,1]
            Q_new = Q2_base + dt * 0.5 * dQ_2
            dp_st = byr_st[nn,2] * Q_new + byr_st[nn,0] * Q_new * abs(Q_new)
            dQ_3 = (P1_new - dp_st - P2_new) * byr_st[nn,1]
            Q_new = Q2_base + dt * dQ_3
            dp_st = byr_st[nn,2] * Q_new + byr_st[nn,0] * Q_new * abs(Q_new)
            dQ_4 = (P1_new - dp_st - P2_new) * byr_st[nn,1]
            Q_new = Q2_base + dt / 6.0 * (dQ_1 + 2.0 * dQ_2 + 2.0 * dQ_3 + dQ_4)

            converge_c1 = Q_new - Qguess1_st

            if abs(Qguess1_st) < 0.00000000001:
                Dif_ratio = 0.0
            else:
                Dif_ratio = abs(converge_c1) / abs(Qguess1_st)

            if Dif_ratio <= converge_cri: # break from the loop if converged
                # if nnn % 200 == 0:
                #     print(f"iter_stn = {iter_stn} at {nnn}")
                break       
            else: # if not converged, update Qguess1_st and repeat the cycle
                Qguess1_st = Qguess1_st + (Q_new - Qguess1_st) * c_relax

        # replace the Q, A, P arterial tree with the new values and finish the calculation
        P1_total = P1_new + 0.5 / roi * ((Q_new / A1_new) ** 2)
        P2_total = P2_new + 0.5 / roi * ((Q_new / A2_new) ** 2)
        DP_total = P1_total - P2_total
        Atreem[n,jstart] = A1_new
        Qtreem[n,jstart] = Q_new
        Utreem[n,jstart] = Qtreem[n,jstart] / Atreem[n,jstart]
        Ptreem[n,jstart] = tbeta[n,jstart] * (sqrt(Atreem[n,jstart] / A0[n,jstart]) - 1.0) + p0
        Atreem[n,jend] = A2_new
        Qtreem[n,jend] = Q_new
        Utreem[n,jend] = Qtreem[n,jend] / Atreem[n,jend]
        Ptreem[n,jend] = tbeta[n,jend] * (sqrt(Atreem[n,jend] / A0[n,jend]) - 1.0) + p0
        Atree[n,jstart] = A1_new
        Qtree[n,jstart] = Q_new
        Utree[n,jstart] = Qtree[n,jstart] / Atree[n,jstart]
        Atree[n,jend] = A2_new
        Qtree[n,jend] = Q_new
        Utree[n,jend] = Qtree[n,jend] / Atree[n,jend]
        sten_len_i = float(jend - jstart - 1)
        
        for j in range(jstart + 1, jend):
            sten_dis = float(j - jstart - 1)
            Atreem[n,j] = Atree[n,j]
            Qtreem[n,j] = Qtreem[n,jstart]
            Utreem[n,j] = Qtreem[n,j] / Atreem[n,j]
            Ptreem[n,j] = P1_total - DP_total / sten_len_i * sten_dis - 0.5 / roi * ((Qtreem[n,j] / Atreem[n,j]) ** 2.0)
        
    # return the updated Atreem, Qtreem, Utreem, Ptreem arrays from stenosis calculation
    return [Atreem, Qtreem, Utreem, Ptreem, Atree, Qtree, Utree] 


# calculate b_st, y_st, r_st for stenosis model (byr_st[n])
C_kt = 1.52 # Kt for blunt plug [Seeley, 1976]
C_ku = 1.2 # Ku [Young, 1973b]
# r_st = Res_stn[n]
# b_st = C_kt / roi / (A0[n,jend] * A0[n,jend]) / 2.0 * ((A0[n,jend] / A0[n,jneck] - 1.0)*(A0[n,jend] / A0[n,jneck] - 1.0))
# y_st = C_ku * stn_len / roi / A0[n,jend]
# @njit(fastmath=True)
def calc_byr_st(Res_stn,byr_st,numb_stn,nst_ran,A0,roi,dx):

    for nn in range(1,numb_stn+1): # loop for each stenosed artery
    
        n = nst_ran[nn,0] # artery no.
        jstart = nst_ran[nn,1] # start node no.
        jend = nst_ran[nn,2] # end node no.
        stn_len = float(jend - jstart) * dx
        jneck = nst_ran[nn,3] # neck node no.

        r_st = Res_stn[n]
        b_st = C_kt / roi / (A0[n,jend] * A0[n,jend]) / 2.0 * ((A0[n,jend] / A0[n,jneck] - 1.0)*(A0[n,jend] / A0[n,jneck] - 1.0))
        y_st = C_ku * stn_len / roi / A0[n,jend]
        inv_y_st = 1.0 / y_st

        byr_st[nn,0] = b_st
        byr_st[nn,1] = inv_y_st
        byr_st[nn,2] = r_st
        
        print(f"Stenosis artery no.{n}: r_st = {r_st:.6e} [Pa.s/m^3], b_st = {b_st:.6e} [Pa.s^2/m^6], y_st = {y_st:.6e} [1/(Pa.s)]")

    return byr_st
