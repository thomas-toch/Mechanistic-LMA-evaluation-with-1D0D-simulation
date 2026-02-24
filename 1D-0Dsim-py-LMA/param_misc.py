from dataclasses import dataclass
import numpy as np
from numba import njit
import struct

@dataclass(frozen=True)
class simparam:
    # max number of nodes, total number of ariteries, and zerod model number
    # Don't change these values
    nartery:int = 83
    nzeromodel:int = 131 # 112 + 7 + 2*6
    numlma:int = 7 # number of LMA nodes
    numcowoutlet:int = 6 # number of CoW outlet arteries
    nzm_terminal:int = 40 # beginning index of terminal artery nodes (40-111)
    nzm_lma:int = 112 # beginning index of LMA nodes (112-)
    nzm_cowoutlet:int = 119 # beginning index of CoW outlet arteries (119-)

    visc_kr:float = 22.0 # Kr for N-S (Plug flow)
    mmhgtoPa:float = 133.3223684 # unit conversion
    
@njit(fastmath=True)
def update_sumtree(Qsumtree, Qtreem, imaxtree, nartery, param):
    for i in range(1, nartery + 1):
        Qsumtree[i] += Qtreem[i,int(imaxtree[i] / 2)] * param

def write_record(file, rec_num, value, recl, fmt='d'):
    """
    指定されたrec位置に、valueを8バイトで書き込む
    fmt: 'd' = float64, 'i' = int32
    """
    file.seek((rec_num - 1) * recl)
    file.write(struct.pack(fmt, value))

radtolen = 2.5
pi = np.pi
mmhgtoPa = simparam.mmhgtoPa

def RLMAcal(frm, rad, numR):
    len = rad * radtolen
    result = 8.0 * frm * len / (pi * (rad ** 4.0)) * 1.0e3 / mmhgtoPa / numR
    return result
        
class DualOutput:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, message):
        for output in self.outputs:
            if not (hasattr(output, 'closed') and output.closed):
                output.write(message)
                output.flush()

    def flush(self):
        for output in self.outputs:
            if not (hasattr(output, 'closed') and output.closed):
                output.flush()

    def close(self):
        for output in self.outputs:
            if hasattr(output, 'close') and not output.closed:
                output.close()

# mbif_par for cow_geo != 0
mbif_par_adjust = np.zeros((11,18,3), dtype=np.int64)
# complete
mbif_par_adjust[0,:,:] = [[1,68,69], # 40
                          [1,59,60], # 47
                          [1,57,71], # 56
                          [2,59,58], # 57
                          [0,0,0], # 58
                          [3,0,0], # 59
                          [1,61,62], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [1,66,67], # 68
                          [3,0,0], # 69
                          [0,0,0], # 70
                          [2,69,70]] # 71
# ACom
mbif_par_adjust[1,:,:] = [[1,68,69], # 40
                          [1,59,60], # 47
                          [1,57,71], # 56
                          [2,59,58], # 57
                          [0,0,0], # 58
                          [3,0,0], # 59
                          [1,61,62], # 60
                          [0,0,0], # 61
                          [4,63,0], # 62
                          [5,62,0], # 63
                          [6,0,0], # 64
                          [5,66,0], # 65
                          [4,65,0], # 66
                          [0,0,0], # 67
                          [1,66,67], # 68
                          [3,0,0], # 69
                          [0,0,0], # 70
                          [2,69,70]] # 71
# Rt. ACA1 
mbif_par_adjust[2,:,:] = [[1,68,69], # 40
                          [1,59,60], # 47
                          [1,57,71], # 56
                          [2,59,58], # 57
                          [0,0,0], # 58
                          [3,0,0], # 59
                          [4,61,0], # 60
                          [5,60,0], # 61
                          [6,0,0], # 62
                          [5,64,0], # 63
                          [4,63,0], # 64
                          [0,0,0], # 65
                          [1,64,65], # 66
                          [0,0,0], # 67
                          [1,66,67], # 68
                          [3,0,0], # 69
                          [0,0,0], # 70
                          [2,69,70]] # 71
# Lt. ACA1
mbif_par_adjust[3,:,:] = [[1,68,69], # 40
                          [1,59,60], # 47
                          [1,57,71], # 56
                          [2,59,58], # 57
                          [0,0,0], # 58
                          [3,0,0], # 59
                          [1,61,62], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [4,65,0], # 64
                          [5,64,0], # 65
                          [6,0,0], # 66
                          [5,68,0], # 67
                          [4,67,0], # 68
                          [3,0,0], # 69
                          [0,0,0], # 70
                          [2,69,70]] # 71
# Rt. PCom
mbif_par_adjust[4,:,:] = [[1,68,69], # 40
                          [1,61,62], # 47
                          [1,57,71], # 56
                          [4,58,0], # 57
                          [5,57,0], # 58
                          [6,0,0], # 59
                          [6,0,0], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [1,66,67], # 68
                          [3,0,0], # 69
                          [0,0,0], # 70
                          [2,69,70]] # 71
# Lt. PCom
mbif_par_adjust[5,:,:] = [[1,66,67], # 40
                          [1,59,60], # 47
                          [1,57,71], # 56
                          [2,59,58], # 57
                          [0,0,0], # 58
                          [3,0,0], # 59
                          [1,61,62], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [6,0,0], # 68
                          [6,0,0], # 69
                          [5,71,0], # 70
                          [4,70,0]] # 71
# Both PComs
mbif_par_adjust[6,:,:] = [[1,66,67], # 40
                          [1,61,62], # 47
                          [1,57,71], # 56
                          [4,58,0], # 57
                          [5,57,0], # 58
                          [6,0,0], # 59
                          [6,0,0], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [6,0,0], # 68
                          [6,0,0], # 69
                          [5,71,0], # 70
                          [4,70,0]] # 71
# Rt. PCA1
mbif_par_adjust[7,:,:] = [[1,68,69], # 40
                          [1,59,60], # 47
                          [1,69,70], # 56
                          [6,0,0], # 57
                          [5,59,0], # 58
                          [4,58,0], # 59
                          [1,61,62], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [1,66,67], # 68
                          [3,0,0], # 69
                          [0,0,0], # 70
                          [6,0,0]] # 71
# Lt. PCA1
mbif_par_adjust[8,:,:] = [[1,68,69], # 40
                          [1,59,60], # 47
                          [1,58,59], # 56
                          [6,0,0], # 57
                          [0,0,0], # 58
                          [3,0,0], # 59
                          [1,61,62], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [1,66,67], # 68
                          [4,70,0], # 69
                          [5,69,0], # 70
                          [6,0,0]] # 71
# Rt.PCom + Lt. PCA1
mbif_par_adjust[9,:,:] = [[1,68,69], # 40
                          [1,61,62], # 47
                          [4,57,58], # 56
                          [6,0,0], # 57
                          [5,56,57], # 58
                          [6,0,0], # 59
                          [6,0,0], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [1,66,67], # 68
                          [4,70,0], # 69
                          [5,69,0], # 70
                          [6,0,0]] # 71
# Lt.PCom + Rt. PCA1
mbif_par_adjust[10,:,:] = [[1,66,67], # 40
                          [1,59,60], # 47
                          [4,71,70], # 56
                          [6,0,0], # 57
                          [5,59,0], # 58
                          [4,58,0], # 59
                          [1,61,62], # 60
                          [0,0,0], # 61
                          [1,63,64], # 62
                          [0,0,0], # 63
                          [2,66,65], # 64
                          [0,0,0], # 65
                          [3,0,0], # 66
                          [0,0,0], # 67
                          [6,0,0], # 68
                          [6,0,0], # 69
                          [5,56,71], # 70
                          [6,0,0]] # 71

# # R-resistance, Y-inertance, E-elastance, C-compliance
# Rzd = np.array(21, dtype=np.float64) # correspond to Q node
# Czd = np.array(20, dtype=np.float64) # correspond to V node
# Yzd = np.array(20, dtype=np.float64) # correspond to Q node
# Szd = np.array(21, dtype=np.float64) # correspond to V node

# # correspond to V node
# Szd = [ 
#     Sra, # atrium and ventricle maybe reversed?
#     Srv, # variable
#     Spua,
#     Spuc,
#     Spuv,
#     Sla, # atrium and ventricle maybe reversed?
#     Slv, # variable
#     S_vcu,
#     S_vcuu,
#     S_upperv,
#     S_upperc,
#     S_upper,
#     S_vcl,
#     S_vclu,
#     S_lowerv,
#     S_lowerc,
#     S_lower,
#     Spwa,
#     Spwc,
#     Spwv,
#     Saa
# ]

# # correspond to V node
# Czd = [ 
#     Crv, # non-existent
#     Cra,
#     Cpua,
#     Cpuc,
#     Cpuv,
#     Clv,
#     Cla, # non-existent
#     C_vcu,
#     C_vcuu,
#     C_upperv,
#     C_upperc,
#     C_upper,
#     C_vcl,
#     C_vclu,
#     C_lowerv,
#     C_lowerc,
#     C_lower,
#     Cpwa, # non-existent
#     Cpwc,
#     Cpwv # non-existent
# ]

# # correspond to Q node
# Rzd = [ 
#     Rtv,
#     Rpv,
#     Rpua,
#     Rpuc,
#     Rpuv,
#     Rmv,
#     Rav,
#     R_vcu1,
#     R_vcu,
#     R_vcuu,
#     R_upperc, # R_upperv?
#     R_upperc,
#     R_vcl1,
#     R_vcl,
#     R_vclu,
#     R_lowerc, # R_lowerv?
#     R_lowerc,
#     gpw, # 1/Rpwa
#     Rpwc,
#     Rpwv,
#     Ra
# ]

# # correspond to Q node
# Yzd = [ 
#     ytv,
#     ypv,
#     ypua,
#     ypuc,
#     ypuv,
#     ymv,
#     yav,
#     y_vcu1,
#     y_vcu,
#     y_vcuu,
#     y_upperv,
#     y_upperc,
#     y_vcl1,
#     y_vcl,
#     y_vclu,
#     y_lowerv,
#     y_lowerc,
#     ypwa,
#     ypwc,
#     ypwv
# ]

# Zzd = np.zeros(6, dtype=np.float64) # correspond to V node

# Zzd = [
#     Zpua,
#     Zpuc,
#     Zpuv,
#     Zpwc,
#     Zpwv,
#     Zaa
# ]

# Bvalve = np.zeros(4, dtype=np.float64) # Bernoulli's resistance of valves

# Bvalve = [
#     bav, # aortic valve
#     bmv, # mitral valve
#     btv, # tricuspid valve
#     bpv  # pulmonary valve
# ]

# # variables
# Ezd = np.zerod(21, dtype=np.float64) # E(0-7,18-19,aa)
# Astate = np.zerod(4, dtype=np.float64) # a(valve state)
# pzd = np.zerod(4, dtype=np.float64) # ppp,pit(const),plv,prv,FL,FR1
# # S(0,1,5,6): update Szd

# # other const
# # Vmax,Es,Sva0,Vpe,Vpc0,Vcon # for updatecat1
# # Eaa0, Epua0, Epuc0, Epuv0, Epwc0, Epwv0 # for updatecat2(Ezd)
# # cardiac chamber parameters Ea,Eb total 8 parameters # for ecals