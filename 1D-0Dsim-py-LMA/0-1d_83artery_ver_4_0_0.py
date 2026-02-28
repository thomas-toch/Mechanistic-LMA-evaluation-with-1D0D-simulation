'''
ORIGINAL CODE:
0-1D HEMODYNAMIC MODEL OF THE HUMAN CARDIOVASCUALR SYSTEM         
WITH THE 71 ARTERY TREE REPRESENTED BY A 1D MODEL AND THE         
REMAINDER DESCRIBED BY A 0D MODEL                                 
                                                                
NUMERICAL METHOD:                                                 
    TWO-STEP LAX-WENDROFF METHOD FOR THE 1D MODEL                     
    FOURTH ORDER RUNGE-KUTTA METHOD FOR THE 0D MODEL                  
                                                                
                        COPYRIGHT@ FUYOU LIANG   RIKEN  2007.10  

CURRENT VERSION
    VERSION 4.0.0 2025.04 by Kotani   
        Moved code to Python
        Partially modularized   
        Output parameters adjusted                            
PAST FORTRAN VERSIONS                                            
    VERSION 3.1.3 2020.01.23                                         
        Bessems' stenosis model revised (uses Res_stn)                 
        Debugged for multiple stenoses (deallocation of r_temp)        
        Debugged for direct use of stenosis & distal diameters         
        Virtual surgery routine revised (case of vsurg=0)              
        Output stenosis parameters (Rv, Dn, SR)                        
        Output adjusted parameters (CoW PRs, PR total scale, mean BP)  
    VERSION 3.1.2 2019.12.02                                         
        Bessems' stenosis model & surgery prediction implemented       
    VERSION 3.1.1 2019.07.13                                         
        83-artery baseline model                                       
    VERSION 3.1.0 2019.07.07                                         
        Code is rearranged for calculation speed improvement           
    VERSION 3.0.0 2017.07.24   
'''

# import modules
import numpy as np
import csv
import math 
from numba import njit
from datetime import datetime
import os
import sys
from scipy.optimize import minimize

# parameter/module initialization
import zerod as zd # import parameters and functions for 1D interface
import bifurcation as bif # import functions for bifurcation computation
import laxwendroff as lw # import functions for Lax-Wendroff scheme
import stenosis as stn # import functions for stenosis calculation
import param_misc as pm # import calculation parameters
import tbeta as tb # import tbeta calculation function

sp = pm.simparam
nartery = sp.nartery  # number of arteries
nzeromodel = sp.nzeromodel # number of 0d models
numlma = sp.numlma # number of LMA nodes
numcowoutlet = sp.numcowoutlet # number of CoW outlet arteries
nzm_terminal = sp.nzm_terminal # beginning index of terminal artery nodes (40-111)
nzm_lma = sp.nzm_lma # beginning index of LMA nodes (112-)
nzm_cowoutlet = sp.nzm_cowoutlet # beginning index of CoW outlet arteries (119-)
visc_kr = sp.visc_kr  # Kr for N-S (Plug flow)
mmhgtoPa = sp.mmhgtoPa  # mmHg to Pa conversion factor

# calculations
sqrt = np.sqrt
exp = np.exp
log = np.log
mod = np.mod
pi = np.pi

# initializing log.txt
DualOutput = pm.DualOutput # import DualOutput class from param_misc.py

# Create output directory
# Initialize output file name
now = datetime.now()
filename_output = now.strftime("output_%Y%m%d_%H%M%S/")
output_directory = filename_output
# output_directory = "output/" # change this if you want your desired output directory name
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Output directory '{output_directory}' created.")
else:
    print(f"Output directory '{output_directory}' already exists.")

# initialize log file
log_file_name = filename_output + 'log.txt'
log_file = open(log_file_name, 'w', encoding='utf-8')

# 標準出力とファイルの両方に出力
sys.stdout = DualOutput(sys.stdout, log_file)
print(f"Log file initialized:{log_file_name}")

# set input file names
print("Read calculation parameters.")
# file name settings
folderCSV = "input_csv/"
arteryCSV = "Artery.csv"
calParamCSV = "CalParam.csv"
CoWmeasuredCSV = "CoW_measured.csv"
stenosisCSV = "Stenosis.csv"
PS_geometryCSV = "PS_geometry.csv"
R_comCSV = "R_com_adjusted.csv"

#computaion conditions (CalParam.csv)
file_path = folderCSV + calParamCSV
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = [next(reader) for _ in range(4)]  # skip
    dx = float(next(reader)[1]) # grid size [m]
    crno = float(next(reader)[1]) # courant number
    c_wave = float(next(reader)[1]) # reference wave speed [m/s]
    Tduration = float(next(reader)[1]) # cardiac duration [s]
    nofduration = int(next(reader)[1]) # No. of cardiac cycles to calculate; !!!set to even no. of cycles!!!, 0: No,1: Yes
    nprint = int(next(reader)[1]) # time step interval of outputing results
    init_1d = int(next(reader)[1]) # initialize Q,A,P with previous results?, 0: No,1: Yes (1DIC_QAP.dat)
    init_mend_stn = int(next(reader)[1]) # Initialize with stenosis virtual surgery? (make sure vsurg_stn=0), 0: No,1: Yes
    init_0d = int(next(reader)[1]) # Initialize 0D model with old results?,0: No,1: Yes
    init_peri_params = int(next(reader)[1]) # Initialize peripheral parameters with previous results?,0: No,1: Yes
    mstn = int(next(reader)[1]) # Calculate stenosis?,0: No,1: Yes
    geometry = int(next(reader)[1]) # Type of arterial geometry of CoW,0: Original,1: Original P-S,2: Detailed P-S
    converge_err = float(next(reader)[1]) # convergence error threshold
    converge_err_Parm = float(next(reader)[1]) # Parm convergence error threshold
    converge_err_cereb = float(next(reader)[1]) # Cow convergence error threshold
    converge_err_LMA = float(next(reader)[1]) # LMA convergence error threshold
    alpha = float(next(reader)[1]) # regulation coefficient
    alpha_4DFlow = float(next(reader)[1]) # regulation coefficient for R1,R2:4D flow; for PR_reg_LMA=1
    alpha_SPECT = float(next(reader)[1]) # regulation coefficient for R3:SPECT; for PR_reg_LMA=1
    alpha_LMA = float(next(reader)[1]) # regulation coefficient for RLMA; for PR_reg_LMA=1
    visualization = int(next(reader)[1]) # Visualize results?,0: No,1: Yes
    viz_str = int(next(reader)[1]) # visualization start cardiac cycle
    viz_end = int(next(reader)[1]) # visualization end cardiac cycle
    PR_reg_total = int(next(reader)[1]) # Regulate total PRs using patient's blood pressure?,0: No,1: Yes
    PR_reg_cereb = int(next(reader)[1]) # Regulate cerebral PR?,0: No,1: Yes
    PR_reg_LMA = int(next(reader)[1]) # Regulate LMA R?, 0: No,1: Yes
    r_com_reg = int(next(reader)[1]) # Regulate radii of communicating arteries?,0: No,1: Yes
    r_com_read = int(next(reader)[1]) # Read radii of communicating arteries from external file?,0: No,1: Yes
    vsurg_stn = int(next(reader)[1]) # Simulate a virtual surgery and compute virtual post-sugery results?,0: No,1: Yes(Single-stenosis Case or Double-stenosis Case;Both),2: Yes(Double-stenosis Case;L>R),3: Yes(Double-stenosis Case;R>L)
    vsurg_stn_qap_init = int(next(reader)[1]) # Initialize virtual surgery with previous results?,0: No,1: Yes
    sim_LMA = int(next(reader)[1]) # Simulate 0d model with Couple-Lumped/Detailed LMA models?,0: No,1: Yes;Lumped,2: Yes;Detailed
    RLMA_calc = int(next(reader)[1]) # Method of calculating RLMA?,0: Set values from Artery.csv,1: Use Thomas vasculature
    temp = next(reader)
    cow_geo = int(temp[1]) # Type of CoW geometry,0: Complete,1: ACom,2: Rt. ACA1,3: Lt.ACA1,4: Rt. Pcom,5: Lt. PCom,6: Both PComs,7: Rt. PCA1,8: Lt. PCA1,9: Rt. PCom/Lt. PCA1,10: Lt.PCom/Rt. PCA1
    cow_geo_name = temp[cow_geo+4] # Name of CoW geometry
    adjust_stn_effect = int(next(reader)[1]) # Adjust stenosis effect?,0: No,1: Yes(Emphasize)
    simplified_output = int(next(reader)[1]) # Simplifiy output files? (Only will output mean data stc.),0: No,1: Yes
    headers = [next(reader) for _ in range(3)] # skip
    ro = float(next(reader)[1]) #  density [kg/m^3]
    frm = float(next(reader)[1]) # viscosity coefficient [Pa.s]
    age = float(next(reader)[1]) # age
    p0mmhg = float(next(reader)[1]) # pressure to start calculation [mmHg]
    Parm_ref = float(next(reader)[1]) # Reference blood pressure in arm (Highest*(1/3) + Lowest*(2/3)) [mmHg]
    
dt = float(dx) * float(crno) / float(c_wave)  # dt = dx * crno / c_wave
dxi = 1.0 / float(dx)  # inverse of dx
fr = frm / ro  # dynamic viscosity coefficient
roi = 1.0 / float(ro)  # inverse of density
stop_flag = 0 # stop flag for the calculation

# parameter adjustments
if nofduration % 2 == 1:
    nofduration += 1
    print(f"Number of cardiac cycles is adjusted to {nofduration}; a even number.")

# print parameters
print ("Read calculation parameters.")
print ("dx = ", dx)
print ("dt = ", dt)
print ("crno = ", crno)
print ("c_wave = ", c_wave)
print ("Tduration = ", Tduration)
print ("nofduration = " , nofduration)
print ("nprint = ", nprint)
print ("init_1d = ", init_1d)
print ("init_mend_stn = ", init_mend_stn)
print ("init_0d = ", init_0d)
print ("init_peri_params = ", init_peri_params)
print ("mstn = ", mstn)
print ("geometry = ", geometry)
print ("converge_err = ", converge_err)
print ("converge_err_Parm = ", converge_err_Parm)
print ("converge_err_cereb = ", converge_err_cereb)
print ("converge_err_LMA = ", converge_err_LMA)
print ("alpha = ", alpha)
print ("alpha_4DFlow = ", alpha_4DFlow)
print ("alpha_SPECT = ", alpha_SPECT)
print ("alpha_LMA = ", alpha_LMA)
print ("visualization = ", visualization)
print ("viz_str = ", viz_str)
print ("viz_end = ", viz_end)
print ("PR_reg_total = ", PR_reg_total)
print ("PR_reg_cereb = ", PR_reg_cereb)
print ("PR_reg_LMA = ", PR_reg_LMA)
print ("r_com_reg = ", r_com_reg)
print ("r_com_read = ", r_com_read)
print ("vsurg_stn = ", vsurg_stn)
print ("vsurg_stn_qap_init = ", vsurg_stn_qap_init)
print ("sim_LMA = ", sim_LMA)
print ("RLMA_calc = ", RLMA_calc)
print ("cow_geo = ", cow_geo)
print ("cow_geo_name = ", cow_geo_name)
print ("adjust_stn_effect = ", adjust_stn_effect)
print ("simplified_output = ", simplified_output)
print ("ro = ", ro)
print ("frm = ", frm)
print ("age = ", age)
print ("p0mmhg = ", p0mmhg)
print ("Parm_ref = ", Parm_ref)

# Literature data of arterial tree (Artery.csv)
file_path = folderCSV + arteryCSV
# array initialization
Dtree = np.zeros(nartery+numlma+1, dtype=np.float64)
Rtree = np.zeros((nartery+numlma+1,2), dtype=np.float64) # proximal and distal radii
RCtree = np.zeros((nartery+numlma+1, 3), dtype=np.float64) # R, L, C
mbif_par = np.zeros((nartery+numlma+1, 3), dtype=np.int64)
artery_name = np.zeros((nartery+numlma+1, 2), dtype=object)

# Load parameters from Artery.csv (Complete CoW)
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    headers = [next(reader) for _ in range(3)]  # skip
    for i in range(1, nartery+numlma+1):
        temp = next(reader)
        arteryno = int(temp[0]) # artery no.
        Dtree[arteryno] = float(temp[1]) # Length of each artery [m]
        Rtree[arteryno,0] = float(temp[2]) # Proximal radii of each artery [m]
        Rtree[arteryno,1] = float(temp[3]) # Distal radii of each artery [m]
        RCtree[arteryno,0] = float(temp[4]) # Total resistance R of WK3 [mmHg.s/ml]
        RCtree[arteryno,1] = float(temp[5]) # Compliance C of WK3 [ml/mmHg]
        RCtree[arteryno,2] = float(temp[6]) # Inductance L of 2nd node [mmHg.s^2/ml]
        mbif_par[arteryno,0] = int(temp[7]) # bifurcation condition 0: terminal end, 1: branching point 2: confluence point 3: PComA
        mbif_par[arteryno,1] = int(temp[8]) # bifurcation artery no.1
        mbif_par[arteryno,2] = int(temp[9]) # bifurcation artery no.2
        artery_name[arteryno,0] = int(temp[0]) 
        artery_name[arteryno,1] = str(temp[10])
        # print(f"{artery_name[i,0]} = {artery_name[i,1]}")
        # print(f"Length = {Dtree[i]}")
        # print(f"Proximal radius = {Rtree[i,0]}")
        # print(f"Distal radius = {Rtree[i,1]}")
        # print(f"Total resistance = {RCtree[i,0]}")
        # print(f"Compliance = {RCtree[i,1]}")
        # print(f"Inductance = {RCtree[i,2]}")
        # print(f"Bifurcation condition = {mbif_par[i,0]}")
        # print(f"Bifurcation artery no.1 = {mbif_par[i,1]}")
        # print(f"Bifurcation artery no.2 = {mbif_par[i,2]}")

# for incomplete CoW geometry, change Rtree, Dtree and mbif_par

# define missing arteries for incomplete CoW model
exclude_artery = np.zeros(nartery+1, dtype=np.int64) # artery to exclude in calculation; 1: exclude, 0: include
# mbif_par[n,0] = 4: continuous artery(proximal), mbif_par[n,0] = 5: continuous artery(distal and terminal), mbif_par[n,0] = 6: missing artery 
CoW_arteries = [40,47,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]
if cow_geo != 0:

    # set mbif_par using the data from param_misc.py
    mbif_par_adjust = pm.mbif_par_adjust[cow_geo]
    for i in range(len(mbif_par_adjust)):
        arteryno = CoW_arteries[i]
        mbif_par[arteryno,:] = mbif_par_adjust[i,:]
    
    # adjust Dtree, Rtree, RCtree using the data from param_misc.py
    for i in CoW_arteries:
        if mbif_par[i,0] == 4:
            idx4 = i
            idx5 = mbif_par[i,1]
            idx6 = mbif_par[i,2]
        # set Dtree
        Dtree[i] += Dtree[idx5]
        if idx6 != 0:
            Dtree[i] += Dtree[idx6]
        # set Rtree (only distal radius is needed to change)
        if idx6 == 0:
            Rtree[i,1] = Rtree[idx5,1]
        else:
            Rtree[i,1] = Rtree[idx6,1]
        # set RCtree
        if idx6 == 0:
            RCtree[i,:] = RCtree[idx5,:]
        else:
            RCtree[i,:] = RCtree[idx6,:]
# set arteries to exclude from calculation
for i in range(1, nartery+1):
    if mbif_par[i,0] in [5,6]: 
        exclude_artery[i] = 1
# adjust zd.arterynode
for i in range(len(zd.arterynode)):
    if zd.arterynode[i,0] < nartery+1:
        if mbif_par[zd.arterynode[i,0],0] == 5:
            zd.arterynode[i,0] = mbif_par[zd.arterynode[i,0],1] # replace missing artery no. with proximal artery no.

# for debug
# if np.any(Rtree == 0.0):
#     raise ValueError("Rtree にゼロが含まれています")

print("Read literature data of arterial tree.")

# Reference CoW flow rates (CoW_measured.csv)
file_path = folderCSV + CoWmeasuredCSV
# array initialization
n_CoW = np.array([58,61,63,65,67,70,40,47,56], dtype=np.int64) # adjusted order
n_CoW_rev = np.full(nartery+1, -1, dtype=np.int64) # reverse index of n_CoW
Qref = np.zeros((nartery+numlma+1,3), dtype=np.float64) # 0: 4Dflow, 1: SPECT, 2: Calculated LMA
coeff_CoW = np.zeros((nartery+numlma+1,3), dtype=np.float64) # coefficients for CoW regulation
order_CoW = np.zeros((3,3), dtype=np.int64) # order of arteries in CoW outlet pressures (will be defined from LMA flow rates)

# if geometry is incomplete CoW, adjust n_CoW
if cow_geo != 0:
    for i in range(len(n_CoW)):
        if mbif_par[n_CoW[i],0] == 5:
            n_CoW[i] = mbif_par[n_CoW[i],1] # replace missing artery no. with proximal artery no.

print(f"n_CoW = {n_CoW}")

for i in range(0,9):
    n_CoW_rev[n_CoW[i]] = i

# if geometry is incomplete CoW, adjust zd.LMA_connection
if cow_geo != 0:
    for i in range(len(zd.LMA_connection)):
        for j in range(2):
            if mbif_par[zd.LMA_connection[i,j],0] == 5:
                zd.LMA_connection[i,j] = mbif_par[zd.LMA_connection[i,j],1] # replace missing artery no. with proximal artery no.

print("Read Reference CoW flow rates.")
print("Reference CoW flow rates [ml/min]:")
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = [next(reader) for _ in range(2)]  # skip
    for i in range(22):
        temp = next(reader)
        ndummy = int(temp[0])
        arteryno = int(temp[1])
        index = int(temp[3])
        Qref[arteryno,index] = float(temp[4]) / 60.0 # convert to [ml/s]

# adjust Qref for incomplete CoW
if cow_geo != 0:
    for i in range(1,nartery+numlma+1):
        if mbif_par[i,0] == 5:
            Qref[mbif_par[i,1],0] = Qref[i,0]
            Qref[mbif_par[i,1],1] = Qref[i,1]
            Qref[mbif_par[i,1],2] = Qref[i,2]

# define order_CoW
if Qref[87,2] >= 0.0:
    order_CoW[0,:] = [65,63,0] # 65 > 63
else:
    order_CoW[0,:] = [63,65,0] # 63 > 65
if Qref[84,2] >= 0.0:
    if Qref[85,2] < 0.0 and Qref[86,2] >= 0.0:
        order_CoW[1,:] = [58,63,61] # 58 > 63 > 61
    elif Qref[85,2] >= 0.0 and Qref[86,2] < 0.0:
        order_CoW[1,:] = [61,58,63] # 61 > 58 > 63
    elif Qref[85,2] < 0.0 and Qref[86,2] < 0.0:
        order_CoW[1,:] = [58,61,63] # 58 > 61 > 63
    else:
        order_CoW[1,:] = [0,0,0] # unknown
elif Qref[84,2] < 0.0:
    if Qref[85,2] >= 0.0 and Qref[86,2] < 0.0:
        order_CoW[1,:] = [61,63,58] # 61 > 63 > 58
    elif Qref[85,2] < 0.0 and Qref[86,2] >= 0.0:
        order_CoW[1,:] = [63,58,61] # 63 > 58 > 61
    elif Qref[85,2] >= 0.0 and Qref[86,2] >= 0.0:
        order_CoW[1,:] = [63,61,58] # 63 > 61 > 58
    else:
        order_CoW[1,:] = [0,0,0] # unknown
if Qref[88,2] >= 0.0:
    if Qref[89,2] < 0.0 and Qref[90,2] >= 0.0:
        order_CoW[2,:] = [67,65,70] # 67 > 65 > 70
    elif Qref[89,2] >= 0.0 and Qref[90,2] < 0.0:
        order_CoW[2,:] = [70,67,65] # 70 > 67 > 65
    elif Qref[89,2] < 0.0 and Qref[90,2] < 0.0:
        order_CoW[2,:] = [67,70,65] # 67 > 70 > 65
    else:
        order_CoW[2,:] = [0,0,0] # unknown
elif Qref[88,2] < 0.0:
    if Qref[89,2] >= 0.0 and Qref[90,2] < 0.0:
        order_CoW[2,:] = [70,65,67] # 70 > 65 > 67
    elif Qref[89,2] < 0.0 and Qref[90,2] >= 0.0:
        order_CoW[2,:] = [65,67,70] # 65 > 67 > 70
    elif Qref[89,2] >= 0.0 and Qref[90,2] >= 0.0:
        order_CoW[2,:] = [65,70,67] # 65 > 70 > 67
    else:
        order_CoW[2,:] = [0,0,0] # unknown

print(f"order_CoW = {order_CoW}")

# adjust order_Cow for incomplete CoW
for i in range(3):
    for j in range(3):
        if order_CoW[i,j] != 0 and mbif_par[order_CoW[i,j],0] == 5:
            order_CoW[i,j] = mbif_par[order_CoW[i,j],1]

# calculation of the effect of aging

# radius coefficient
r_con = 0.0097 * 25 + 0.98 # radius of 25 years old
r_age = 0.0097 * age + 0.98 # radius of age years old
c_rad_aor = r_age / r_con # radius coefficient of aorta
c_rad_ela = (c_rad_aor - 1.0) * 1.0 + 1.0 # radius coefficient for some arteries
c_rad_per = 1.0 # radius coefficient of peripheral arteries

# calculation of radius coefficient of each artery
cof_rad = np.zeros(nartery+1, dtype=np.float64) # radius coefficient of each artery
for i in range(1,nartery+1):
    if i in [1,2,10,12,13,25,27,29,31,33]: # aorta trunk
        cof_rad[i] = 1.0 * c_rad_aor 
    elif i in [3,5,11]: # cerebral circulation 1
        cof_rad[i] = 1.0 * c_rad_ela
    elif i in [39,48,47,40]: # cerebral circulation 2
        cof_rad[i] = 1.0 * c_rad_per
    elif i in [4,15]: # upper limb circulation 1
        cof_rad[i] = 1.0 * c_rad_ela
    elif i in [7,17]: # upper limb circulation 2
        cof_rad[i] = 1.0 * c_rad_per
    elif i in [6,16,8,19,9,18,43,46,44,45]: # upper limb circulation 3
        cof_rad[i] = 1.0
    elif i in [14,20,21,22,23,24,26,28,30,32]: # splanic and renal circulation
        cof_rad[i] = 1.0 * c_rad_ela
    elif i in [34,49,35,50]: # lower limb circulation 1
        cof_rad[i] = 1.0 * c_rad_ela
    elif i in [37,52,38,53]: # lower limb circulation 2
        cof_rad[i] = 1.0 * c_rad_per
    elif i in [36,51,41,54,42,55]: # lower limb circulation 3
        cof_rad[i] = 1.0
    else: # for other terminal arteries (i>=56)
        cof_rad[i] = 1.0 

# calculation of radius expansion of each artery
for i in range(1,nartery+1):
    for j in [0,1]:
        Rtree[i,j] = Rtree[i,j] * cof_rad[i]

# elastance coefficient
e_con = -0.017 * 25.0 + 0.001 * 25 * 25 + 5.490 # elastance of 25 years old
e_age = -0.017 * age + 0.001 * age * age + 5.490 # elastance of age years old
c_ela_aor = (e_age / e_con) ** 2.0 # elastance coefficient of aorta

# resistance coefficient
if age <= 50.0:
    c_res = (age - 25.0) / 25.0 * 0.09 + 1 # resistance coefficient of aorta
else:
    c_res = 1.09

print(f"Aging parameters c_ela_aor, c_res = {c_ela_aor}, {c_res}")
print("Calculated the effect of aging.")

# Read patient-specific geometry of vasculature (PS_geometry.csv)

r_temp = [] #temporary array for arterial geometry

if (geometry==1 or geometry==2): # read P-S geometry into temp file
    file_path = folderCSV + PS_geometryCSV
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        headers = [next(reader) for _ in range(1)] #sskip
        for temp in reader:
            r_temp.append(temp)
    
    # for debug
    #print(r_temp)
    #print(len(r_temp))

    # read length and proximal/distal radius of each artery
    for i in range(0,len(r_temp)):
        arteryno = int(r_temp[i][0])
        Dtree[arteryno] = 1.0e-3 * float(r_temp[i][2])
        Rtree[arteryno,0] = 1.0e-3 * float(r_temp[i][4])
        Rtree[arteryno,1] = 1.0e-3 * float(r_temp[i][int(r_temp[i][1]) + 1])

print("Read patient-specific geometry of CoW.")

# calculate the cell number of each artery
imaxtree = np.zeros(nartery+1, dtype=np.int64) # maximum number of nodes
imax_83 = 1
for i in range(1, nartery+1):
    imaxtree[i] = int(Dtree[i] * dxi) # number of nodes for each artery
    if imax_83 < imaxtree[i]:
        imax_83 = imaxtree[i]
    #print(f"imaxtree for artery no. {i} = {imaxtree[i]}")

# calculate the cross-sectional area of each artery and each node
# array initialization
Rtree0 = np.zeros((nartery+1,imax_83+1), dtype=np.float64, order='C') # initial radius of each artery and each node
Atree = np.zeros((nartery+1,imax_83+1), dtype=np.float64, order='C') # cross-sectional area of each artery and each node
A0 = np.zeros((nartery+1,imax_83+1), dtype=np.float64, order='C') # initial cross-sectional area of each artery and each node
# geometry generation for geometry=0,1(exponential geometry)
for i in range(1, nartery+1):
    if (cow_geo != 0 and exclude_artery[i] == 1):
        continue # skip excluded arteries
    for j in range(0, imaxtree[i]+1):
        Rtree0[i,j] = Rtree[i,0] * exp(log(Rtree[i,1] / Rtree[i,0]) * (dx * float(j)) / Dtree[i] )
        Atree[i,j] = pi * (Rtree0[i,j] ** 2.0)
        A0[i,j] = Atree[i,j]
    #print(f"cross-sectional area for artery no. {i} = {A0[i,:]}")
    #print(f"cross-sectional area for artery no. {i}, node {imaxtree[i]} = {A0[i,imaxtree[i]]}")
# for geometry=2 (detailed P-S geometry), replace the radius and cross-sectional area for segmented arteries
if geometry == 2:
    for k in range(0, len(r_temp)):
        i = int(r_temp[k][0])
        if (cow_geo != 0 and exclude_artery[i] == 1):
            continue
        # up until the last node
        # for j in range(0, imaxtree[i]):
        #     i_temp = int((int(r_temp[k][1]) - 2) / imaxtree[i] * j) + 4
        #     Rtree0[i,j] = 1.0e-3 * float(r_temp[k][i_temp])
        #     Atree[i,j] = pi * (Rtree0[i,j] ** 2.0)
        #     A0[i,j] = Atree[i,j]
        # # for the last node
        # Rtree0[i,imaxtree[i]] = 1.0e-3 * float(r_temp[k][int(r_temp[k][1]) + 1])
        # Atree[i,imaxtree[i]] = pi * (Rtree0[i,imaxtree[i]] ** 2.0)
        # A0[i,imaxtree[i]] = Atree[i,imaxtree[i]]

        # new code
        for j in range(0, imaxtree[i]+1):
            i_temp = int((int(r_temp[k][1]) - 3) / imaxtree[i] * j) + 4
            Rtree0[i,j] = 1.0e-3 * float(r_temp[k][i_temp])
            Atree[i,j] = pi * (Rtree0[i,j] ** 2.0)
            A0[i,j] = Atree[i,j]        

        for j in range(imaxtree[i]+1, imax_83+1): # delete the data beyond node no.imaxtree[j]: maybe not necessary
            Rtree0[i,j] = 0.0
            Atree[i,j] = 0.0
            A0[i,j] = 0.0
        #print(f"updated cross-sectional area for artery no. {i} = {A0[i,:]}")
        #print(f"cross-sectional area for artery no. {i}, node {imaxtree[i]} = {A0[i,imaxtree[i]]}")

# for debug
# if np.any(A0 == 0.0):
#     raise ValueError("A0 にゼロが含まれています")

print("Calculated the cross-sectional area of each artery.")

# Read adjusted radii of communicating arteries (R_com_adjusted.csv)

if r_com_read == 1: # read
    r_temp_rcom = [] #temporary array for R_com_adjusted.csv
    file_path = folderCSV + R_comCSV
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        headers = [next(reader) for _ in range(1)] # skip
        for temp in reader:
            r_temp_rcom.append(temp)

    for k in range(0,3):
        i = int(r_temp_rcom[k][0])
        for j in range(0, imaxtree[i]+1):
            Rtree0[i,j] = float(r_temp_rcom[k][j+1])
            Atree[i,j] = pi * (Rtree0[i,j] ** 2.0)
            A0[i,j] = Atree[i,j]
    
    print("Read adjusted radii of communicating arteries from R_com_adjusted.csv. (r_com_read=1)")

else: # don't read
    print("Didn't read adjusted radii of communicating arteries from R_com_adjusted.csv. Use default values. (r_com_read=0)")

# Read stenosis parameters (Stenosis.csv)

#initialize temporal arrays
j_stnTb = np.zeros((nartery+1), dtype=np.int64) # Artery no. of stenosed artery
nst_strTb = np.zeros((nartery+1), dtype=np.int64) # Initial segment no. of stenosed region
nst_endTb = np.zeros((nartery+1), dtype=np.int64) # Terminal segment no. of stenosed region
nst_neckTb  = np.zeros((nartery+1), dtype=np.int64) # Segment no. of stenosed region having minimum radius
redu_ratioTb = np.zeros((nartery+1), dtype=np.float64) # Reduction ratio
mmbypixel = 0.0 # mm per pixel of medical image (read from Stenosis.csv)

#initialize array for stenosis computation
nst_ran = np.zeros((nartery+1,4), dtype=np.int64) # segment number parameters of stenosis
Res_stn = np.zeros((nartery+1), dtype=np.float64) # resistance of stenosis

print("Reading stenosis parameters from Stenosis.csv.")
file_path = folderCSV + stenosisCSV
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    headers = [next(reader) for _ in range(1)] # skip
    numb_stn = np.int64(next(reader)[1]) # number of stenosis
    print(f"Number of stenosis = {numb_stn}")
    headers = [next(reader) for _ in range(1)] # skip
    for i in range(1, numb_stn+1):
        temp = next(reader)
        j_stnTb[i] = np.int64(temp[1])
        nst_strTb[i] = np.int64(temp[2])
        nst_endTb[i] = np.int64(temp[3])
        nst_neckTb[i] = np.int64(temp[4])
        redu_ratioTb[i] = np.float64(temp[5])
        mmbypixel = np.float64(temp[6]) # mm per pixel of medical image
        print(f"Artery no. {j_stnTb[i]}: Stenosis at node no.{nst_strTb[i]}-{nst_endTb[i]}, neck at node no.{nst_neckTb[i]}, reduction ratio = {redu_ratioTb[i]}")

# set stenosis parameters
for i in range(1, numb_stn+1):
    j_stn = j_stnTb[i]

    if geometry == 0: # for geometry using literature data
        redu_ratio = redu_ratioTb[i]
        nst_ran[i,0] = j_stn # artery no.
        nst_ran[i,1] = nst_strTb[i] # starting segment no.
        nst_ran[i,2] = nst_endTb[i] # terminal segment no.
        nst_ran[i,3] = np.int64((nst_strTb[i] + nst_endTb[i]) / 2) # neck segment no.
        print(f"geometry = {geometry}: artery no. {j_stn}, Stenosis at node no.{nst_ran[i,1]}-{nst_ran[i,2]}, neck at node no.{nst_ran[i,3]}, reduction ratio = {redu_ratio}")

    elif geometry == 1 or geometry == 2: # for patient-specific geometry
        # find stenosed artery no. in the data table
        for k in range(0,len(r_temp)):
            if np.int64(r_temp[k][0]) == j_stn:
                i_temp = k
                break
            if k == len(r_temp) - 1 and np.int64(r_temp[k][0] != j_stn):
                print(f"Couldn't find the stenosed artery no.{j_stn} in r_temp[{k}]. There is possibly a mistake in the data table.")
        
        nst_ran[i,0] = j_stn # artery no

        nst_ran[i,1] = int(nst_strTb[i] * imaxtree[j_stn] / int(r_temp[i_temp][1])) # starting segment no.
        nst_ran[i,2] = int(nst_endTb[i] * imaxtree[j_stn] / int(r_temp[i_temp][1])) # terminal segment no.
        nst_ran[i,3] = int(nst_neckTb[i] * imaxtree[j_stn] / int(r_temp[i_temp][1])) # neck segment no.

        # causes bugs? 
        # nst_ran[i,1] = int((nst_strTb[i]-2) * imaxtree[j_stn] / int(int(r_temp[i_temp][1]) - 3)) # starting segment no.
        # nst_ran[i,2] = int((nst_endTb[i]-2) * imaxtree[j_stn] / int(int(r_temp[i_temp][1]) - 3)) # terminal segment no.
        # nst_ran[i,3] = int((nst_neckTb[i]-2) * imaxtree[j_stn] / int(int(r_temp[i_temp][1]) - 3)) # neck segment no.
        # if nst_ran[i,1] < 2:
        #     nst_ran[i,1] = 2

        redu_ratio = redu_ratioTb[i] # reduction ratio (only for geometry=0,1)

        if geometry == 1: # for original geometry
            nst_ran[i,3] = int((nst_ran[i,1] + nst_ran[i,2]) / 2) # neck segment no.
            print(f"geometry = {geometry}: Artery no. {j_stn}, Stenosis at node no.{nst_ran[i,1]}-{nst_ran[i,2]}, neck at node no.{nst_ran[i,3]}, reduction ratio = {redu_ratio}")
        elif geometry == 2: # for detailed P-S geometry
            print(f"geometry = {geometry}: Artery no. {j_stn}, Stenosis at node no.{nst_ran[i,1]}-{nst_ran[i,2]}, neck at node no.{nst_ran[i,3]}, reduction ratio = {redu_ratio}")

if mstn == 1: 

    # set stenosis parameters
    for i in range(1, numb_stn+1):
        j_stn = j_stnTb[i]
        # find stenosed artery no. in the data table
        for k in range(0,len(r_temp)):
            if np.int64(r_temp[k][0]) == j_stn:
                i_temp = k
                break
            if k == len(r_temp) - 1 and np.int64(r_temp[k][0] != j_stn):
                print(f"Couldn't find the stenosed artery no.{j_stn} in r_temp[{k}]. There is possibly a mistake in the data table.")

        if geometry in [1, 2]: 

            if geometry == 2: # for detailed P-S geometry, calculate the resistance of stenosis
                # print(f"geometry = {geometry}: Artery no. {j_stn}, Stenosis at node no.{nst_ran[i,1]}-{nst_ran[i,2]}, neck at node no.{nst_ran[i,3]}, reduction ratio = {redu_ratioTb[i]}")
                print("Calculating the resistance of stenosis for geometry=2.")
                i_str = nst_strTb[i] + 2
                i_end = nst_endTb[i] + 2
                delta_x = 1.0e-3 * float(r_temp[i_temp][2]) / (float(r_temp[i_temp][1]) - 3.0) # m

                A2InvSum = 0.0
                if adjust_stn_effect == 0:
                    for l in range(i_str, i_end+1):
                        A2InvSum += 1.0 / (float(r_temp[i_temp][l]) ** 4.0) # 1/mm^4
                elif adjust_stn_effect == 1:
                    pixel = 2.0
                    print(f"Using pixel = {pixel} for adjusted stenosis resistance calculation.")
                    for l in range(i_str, i_end+1): # calculate Resistance with -pixel radius
                        rstntemp = float(r_temp[i_temp][l]) - pixel * mmbypixel
                        if rstntemp <= 0.0:
                            rstntemp = 1.0e-1 # minimum radius = 0.1mm
                        A2InvSum += 1.0 / ( rstntemp ** 4.0 ) # 1/mm^4
                A2InvSum = A2InvSum * delta_x * 1.0e12 / (pi ** 2.0) # 1/m^3
                Res_stn[j_stn] = 8.0 * pi * fr / roi * A2InvSum # resistance of stenosis, 1/m^3*Pa*s   

                rstncalc = Res_stn[j_stn] * 1.0e-6 / mmhgtoPa # [Pa.s/m^3] to [mmHg.s/ml]
                if rstncalc > 500.0:
                    print(f"Res = {rstncalc}. Setting resistance of stenosis for artery no.{j_stn} to 500 [mmHg.s/ml]")
                    Res_stn[j_stn] = 500.0 * mmhgtoPa * 1.0e6 # back to [Pa.s/m^3]

                print(f"A2InvSum = {A2InvSum}")
                print(f"Artery no:{j_stn},1st term Resistance of stenosis = {Res_stn[j_stn]}")
            
        if geometry in [0, 1]: # create stenosis geometry and calculate R for original geometry (geometry=0,1)
            for j in range(nst_ran[i,1], nst_ran[i,2] + 1): # stenosed region
                Rtree0[j_stn,j] = Rtree0[j_stn,nst_ran[i,2]] * (1.0 - redu_ratio)
                Atree[j_stn,j] = pi * (Rtree0[j_stn,j] ** 2.0)
                A0[j_stn,j] = Atree[j_stn,j]

            A2InvSum = 0.0
            A2InvSum = 1.0 / (A0[j_stn,nst_ran[i,1] + 1]  ** 2.0) * (nst_ran[i,2] - nst_ran[i,1]) * dx
            Res_stn[j_stn] = 8.0 * pi * fr / roi * A2InvSum # resistance of stenosis
            print(f"1st term Resistance of stenosis = {Res_stn[j_stn]}")

    # calculate b_st, y_st, r_st for stenosis model
    byr_st = np.zeros((nartery+1,3), dtype=np.float64) # b_st, y_st, r_st for stenosis model
    byr_st = stn.calc_byr_st(Res_stn,byr_st,numb_stn,nst_ran,A0,roi,dx)
            
    # Output the parameters for stenosis
    output_file_stenosisparamCSV = "Stenosis_param.csv"
    file_path = output_directory + output_file_stenosisparamCSV
    with open(file_path, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Res_stn", "Normal_diameter_(mm)", "SR_NASCET_(%)"])
        for n in range(1,numb_stn+1):
            j_stn = nst_ran[n,0]
            Out_Res_stn = Res_stn[j_stn] * 1.0e-6 / mmhgtoPa # [Pa.s/m^3] to [mmHg.s/ml]
            Out_D_nor = 2.0 * sqrt(A0[nst_ran[n,0],nst_ran[n,2]] / pi) # end
            Out_D_stn = 2.0 * sqrt(A0[nst_ran[n,0],nst_ran[n,3]] / pi) # neck
            Out_D_nor = Out_D_nor * 1.0e3  # [m] to [mm]
            Out_D_stn = Out_D_stn * 1.0e3  # [m] to [mm]
            Out_SR = (1.0 - Out_D_stn / Out_D_nor) * 100.0  # NASCET[%] 
            writer.writerow([Out_Res_stn, Out_D_nor, Out_SR])
    print(f"Stenosis parameters have been written to:{file_path}")

else: # don't read
    print("Didn't read stenosis parameters from Stenosis.csv = no stenosis.")

# mend stenosis if init_mend_stn == 1
if vsurg_stn == 0: # if vsurg_stn == 0
    if init_mend_stn == 1: # initialize stenosis with virtual surgery
        print(f"Performing virtual surgery (removing stenosis)...")
        # Remove stenosis shape from 1D tree
        # exp for only stenosed area
        for n in range(1, numb_stn+1):
            j_stn = nst_ran[n,0]
            nst_str = nst_ran[n,1]
            nst_end = nst_ran[n,2]
            nst_length = nst_end - nst_str
            dR = Rtree0[j_stn,nst_end] - Rtree0[j_stn,nst_str]
            for i in range(nst_str,nst_end+1):
                Rtree0[j_stn,i] = Rtree0[j_stn,nst_str] * exp(log(Rtree0[j_stn,nst_end] / Rtree0[j_stn,nst_str]) * (float(i) - float(nst_str)) / float(nst_length))
                # Rtree0[j_stn,i] = Rtree0[j_stn,nst_str] + float(i - nst_str) / float(nst_length) * dR # alternative: linear
                A0[j_stn,i] = pi * (Rtree0[j_stn,i]**2.0)
                Atree[j_stn,i] = A0[j_stn,i]
            for j in range(0,imaxtree[j_stn]+1): # print
                print(f"Artery {j_stn}, node {j}, Rtree0: {Rtree0[j_stn,j]:.6f} mm, Atree: {Atree[j_stn,j]:.6f} mm^2")
        # another way: setting stenosed artery to exponential shape between R0 and R1(geometry == 1)
        # for n in range(1, numb_stn+1):
        #     i = nst_ran[n,0]
        #     for j in range(0, imaxtree[i]+1):
        #         Rtree0[i,j] = Rtree[i,0] * exp(log(Rtree[i,1] / Rtree[i,0]) * (dx * float(j)) / Dtree[i] )
        #         Atree[i,j] = pi * (Rtree0[i,j] ** 2.0)
        #         A0[i,j] = Atree[i,j]
        #         print(f"Artery {i}, node {j}, Rtree0: {Rtree0[i,j]:.6f} mm, Atree: {Atree[i,j]:.6f} mm^2")

# read Uncertainty range from UR_CoW.csv
UR_CoW = np.zeros((nartery+1,2), dtype=np.float64) # Uncertainty range of CoW arteries
UR_CoWCSV = "UR_CoW.csv"
file_path = folderCSV + UR_CoWCSV
with open(file_path, mode='r') as file:
    print("Reading Uncertainty range from UR_CoW.csv...")
    reader = csv.reader(file)
    headers = [next(reader) for _ in range(1)] # skip
    for i in range(18):
        temp = next(reader)
        artery_no = int(temp[1])
        UR_CoW[artery_no,0] = float(temp[2]) # min
        UR_CoW[artery_no,1] = float(temp[3]) # max
        print(f"Artery no.{artery_no}: UR_min={UR_CoW[artery_no,0]} mm, UR_max={UR_CoW[artery_no,1]} mm")

# adjust radius of CoW arteries if adjust_stn_effect == 1
if adjust_stn_effect == 1:
    print("Adjusting radius of CoW arteries to reflect stenosis effect...")
    if numb_stn == 0:
        print("No stenosis exists. No adjustment is made.")
    elif numb_stn == 1:
        if nst_ran[1,0] == 40:
            # artery_adj = [62,64,66,69,71]
            artery_adj = [64,69]
        elif nst_ran[1,0] == 47:
            # artery_adj = [62,64,66,57,59]
            artery_adj = [64,59]
    elif numb_stn == 2:
        # artery_adj = [62,64,66,69,71,57,59]
        artery_adj = [64,69,59]
    for i in range(len(artery_adj)):
        artery_no = artery_adj[i]
        UR_min = UR_CoW[artery_no,0] * 1.0e-3 # mm to m
        UR_max = UR_CoW[artery_no,1] * 1.0e-3 # mm to m
        R_new = UR_min
        print(f"Artery no.{artery_no}: R_new={R_new:.6f} m")
        for j in range(0, imaxtree[artery_no]+1):
            Rtree0[artery_no,j] = R_new
            Atree[artery_no,j] = pi * (Rtree0[artery_no,j] ** 2.0)
            A0[artery_no,j] = Atree[artery_no,j]

# Output copy of arterial geometry
output_file_geometrycopyCSV = "Geometry_copy.csv"
file_path = output_directory + output_file_geometrycopyCSV
with open(file_path, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Artery No.","Number of data","Length [mm]","Radii [mm]"])
    for i in range(1, nartery+1):
        row = [i, imaxtree[i], Dtree[i] * 1.0e3]
        row.extend([Rtree0[i,j] * 1.0e3 for j in range(0, imaxtree[i]+1)])
        writer.writerow(row)
print(f"Arterial geometry has been written into {file_path}")

# Calculate the Young's modulus of the network

# array initialization
tbeta = np.zeros((nartery+1,imax_83+1), dtype=np.float64, order='C') # tbeta(i,j):Eh/(r0(1-ξ^2)) of each artery and each node
cof_ela = np.full(nartery+1, 4.0 / 3.0, dtype=np.float64) # 1/(1-ξ^2) of each artery (ξ=1/2)

tbeta, cof_ela, c_ela_aor = tb.tbetacal(tbeta, cof_ela, Rtree0, imaxtree, nartery, age, roi, cow_geo, exclude_artery, c_ela_aor)

# Adjust peripheral resistance to ensure nonreflection on wave before C
# Define R, L, C values at 0-1D interface

# parameter initialization
R_total = 0.0 # total resistance of each artery
R_treetotal = 0.0 # total resistance of the arterial tree (all arteries): inverse (1/Rtotal = 1/R1 + 1/R2 + ...)
C_treetotal = 0.0 # total capacitance of the arterial tree (all arteries)
RLCtree = np.zeros((4,nartery+numlma+1,6), dtype=np.float64) # R,L,C of each artery
# (1,j,2):1/L of the second node
# (2,j,1):R of the first node
# (2,j,2):R of the second node
# (3,j,1):1/C of the first node
# (3,j,2):1/C of the second node

# calculate R,L,C for each 0-1D interface artery
for j in range(1,nartery+1):
    if j == 1: # aorta inlet
        i = 0
        c0_1 = sqrt(tbeta[j,i] * roi / 2.0 / sqrt(A0[j,i])) * (A0[j,i] ** 0.25) # c = sqrt(tbeta/2/ρ/sqrt(A0))*A^0.25, tbeta = Eh/(r0(1-ξ^2)), beta = sqrt(pi)*Eh/(1-ξ^2)
        R1_modify = c0_1 / (A0[j,i] * roi)
        R1_modify = R1_modify * 1.0e-6 / mmhgtoPa # [Pa.s/m^3] to [mmHg.s/ml]
        RLCtree[2,j,1] = R1_modify # R of the first node
    if mbif_par[j,0] in [0,4]: # terminal end
        i = imaxtree[j]
        c0_1 = sqrt(tbeta[j,i] * roi / 2.0 / sqrt(A0[j,i])) * (A0[j,i] ** 0.25)
        R1_modify = c0_1 / (A0[j,i] * roi)
        R1_modify = R1_modify * 1.0e-6 / mmhgtoPa # [Pa.s/m^3] to [mmHg.s/ml]
        R_total = RCtree[j,0] # total resistance of wk3 (literature data)
        if j < 58:
            RLCtree[2,j,1] = R1_modify # R of the first node: characreristic impedance of the terminal segment Z0 = ρ*c0/A0 [mmHg.s/ml]
            RLCtree[2,j,2] = R_total - R1_modify # R of the second node : R_total - Z0 [mmHg.s/ml]
            RLCtree[3,j,1] = 1.0 / RCtree[j,1] # 1 / C of the first node [ml/mmHg]
            RLCtree[1,j,2] = 1.0 / RCtree[j,2] # 1 / L of the second node [mmHg.s^2/ml]
        else:
            RLCtree[2,j,1] = R_total * 0.2 # R of the first node : 1/5*R_total (because Z0>R_total) [mmHg.s/ml]
            RLCtree[2,j,2] = R_total * 0.8 # R of the second node [mmHg.s/ml]
            RLCtree[3,j,1] = 1.0 / RCtree[j,1] # 1 / C of the first node [ml/mmHg]
            RLCtree[1,j,2] = 1.0 / RCtree[j,2] # 1 / L of the second node [mmHg.s^2/ml]
            
# set R, L, C for LMA arteries: when importing from artery.csv
for j in range(nartery+1, nartery+numlma+1):
    RLCtree[2,j,1] = RCtree[j,0] # R of the first node [mmHg.s/ml]
    RLCtree[1,j,2] = 1.0 / RCtree[j,2] # 1 / L of the second node [mmHg.s^2/ml]

# adjust R1,R2,R3 for Cow arteries if sim_LMA == 1
if sim_LMA == 1:
    for i in range(6):
        arteryno = n_CoW[i]
        R_total = RCtree[arteryno,0] # total resistance of wk3 (literature data)
        RLCtree[2,arteryno,3] = R_total * 0.8 # R3 = 1/5*R_total
        R_total_wk = R_total - RLCtree[2,arteryno,3] # total resistance of wk3 - R3
        RLCtree[2,arteryno,1] = R_total_wk * 0.2 # R1 = 1/5*RT_wk
        RLCtree[2,arteryno,2] = R_total_wk * 0.8 # R2 = 4/5*RT_wk
        print(f"Initial R of the third node of artery no.{arteryno} R1 = {RLCtree[2,arteryno,1]}, R2 = {RLCtree[2,arteryno,2]}, R3 = {RLCtree[2,arteryno,3]} mmHg.s/ml")

# set params from PPIC.csv if ppic_peripheral_params == 1
if init_peri_params == 1:
    print("Setting peripheral parameters from PPIC.csv")
    file_path = folderCSV + "PPIC.csv"
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        headers = [next(reader) for _ in range(1)] # skip
        for temp in reader:
            j = int(temp[0]) # artery no.
            RLCtree[2,j,1] = float(temp[1]) # R1 [mmHg.s/ml]
            RLCtree[2,j,2] = float(temp[2]) # R2 [mmHg.s/ml]
            RLCtree[2,j,3] = float(temp[3]) # R3 [mmHg.s/ml]
            RLCtree[3,j,1] = 1.0 / float(temp[4]) # 1 / C [ml/mmHg]
            RLCtree[1,j,2] = 1.0 / float(temp[5]) # 1 / L [mmHg.s^2/ml]
            print(f"Artery no.{j}: R1 = {RLCtree[2,j,1]}, R2 = {RLCtree[2,j,2]}, R3 = {RLCtree[2,j,3]}, C = {1.0 / RLCtree[3,j,1]}, L = {1.0 / RLCtree[1,j,2]}")
    print("Peripheral parameters have been set from PPIC.csv")

for j in range(1,nartery+1):
    if mbif_par[j,0] in [0,4]: # terminal end
        R_treetotal += 1.0 / (RLCtree[2,j,1] + RLCtree[2,j,2])
        C_treetotal += 1.0 /RLCtree[3,j,1] 

# print(f"R_treetotalinv = {R_treetotal}")
print(f"R_treetotal = {1.0 / R_treetotal}, C_treetotal = {C_treetotal}")

# Output peripheral parameters
output_file_initialperipheralparamsCSV = "Initial_Peripheral_params.csv"
file_path = output_directory + output_file_initialperipheralparamsCSV
with open(file_path, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Artery No.","PR1 [mmHg*s/ml]","PR2 [mmHg*s/ml]","C [ml/mmHg]","L [mmHg*s^2/ml]"])
    for j in range(1,nartery+1):
        if mbif_par[j,0] in [0,4]:
            row = [j, RLCtree[2,j,1], RLCtree[2,j,2], 1.0 / RLCtree[3,j,1], 1.0 / RLCtree[1,j,2]]
            writer.writerow(row)
    skip_row = ["", "", "", "", ""]
    writer.writerow(skip_row) # skip row
    writer.writerow(["R_total = ", 1.0 / R_treetotal, "C_total = ", C_treetotal, ""])
print(f"Initial peripheral parameters have been written into {file_path}")

# Output file name, path and initialization
# 1d data
# array initialization
Qtree_output = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C')
Utree_output = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C')
Ptree_output = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C')
Atree_output = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C')
# file initialization
if simplified_output == 0:
    Q_1ddat = output_directory + "Q_1d.dat"
    with open(Q_1ddat, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = ["time [s]"]
        for i in range(1, nartery+1):
            row.append(f"{i} {artery_name[i,1]} [ml/s]")
        writer.writerow(row)
    print(f"Flow rate Q at the middle of the artery will be output to {Q_1ddat}.")
    U_1ddat = output_directory + "U_1d.dat"
    with open(U_1ddat, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = ["time [s]"]
        for i in range(1, nartery+1):
            row.append(f"{i} {artery_name[i,1]} [m/s]")
        writer.writerow(row)
    print(f"Velocity U at the middle of the artery will be output to {U_1ddat}.")
    P_1ddat = output_directory + "P_1d.dat"
    with open(P_1ddat, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = ["time [s]"]
        for i in range(1, nartery+1):
            row.append(f"{i} {artery_name[i,1]} [mmHg]")
        writer.writerow(row)
    print(f"Pressure P at the middle of the artery will be output to {P_1ddat}.")
    A_1ddat = output_directory + "A_1d.dat"
    with open(A_1ddat, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = ["time [s]"]
        for i in range(1, nartery+1):
            row.append(f"{i} {artery_name[i,1]} [mm^2]")
        writer.writerow(row)
    print(f"Cross section area A at the middle of the artery will be output to {A_1ddat}.")

Q_1dmeancsv = output_directory + "Q_1dmean.csv"
with open(Q_1dmeancsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no"]
    for i in range(1, nartery+1):
        row.append(f"{i} {artery_name[i,1]} [ml/s]")
    writer.writerow(row)
print(f"Flow rate Q at the middle of the artery, cycle mean will be output to {Q_1dmeancsv}.")
P_1dmeancsv = output_directory + "P_1dmean.csv"
with open(P_1dmeancsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no"]
    for i in range(1, nartery+1):
        row.append(f"{i} {artery_name[i,1]} [mmHg]")
    writer.writerow(row)
print(f"Pressure P at the middle of the artery, cycle mean will be output to {P_1dmeancsv}.")
A_1dmeancsv = output_directory + "A_1dmean.csv"
with open(A_1dmeancsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no"]
    for i in range(1, nartery+1):
        row.append(f"{i} {artery_name[i,1]} [mm^2]")
    writer.writerow(row)
print(f"Cross section area A at the middle of the artery, cycle mean will be output to {A_1dmeancsv}.")

# 0d data
if simplified_output == 0:
    QV_0ddat = output_directory +"QV_0d.dat"
    with open(QV_0ddat, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = ["time [s]"]
        for i in range(0, 20):
            row.append(f"q[{i}]")
            row.append(f"v[{i}]")
        if sim_LMA == 1: # if LMA is simulated
            for i in range(0,numlma):
                row.append(f"Qtree0d[{nartery+1+i}]")
        for i in range(20,56):
            j = zd.lookup_value(zd.arterynode,1,2*i,0)
            row.append(f"Qtree0d[{j}]")
            row.append(f"Vtree0d[{j}]")
        if sim_LMA == 1:
            for i in range(nzm_cowoutlet,nzeromodel,2):
                j = zd.lookup_value(zd.arterynode,1,i,0)
                row.append(f"Qtree0d[{j}]")
                row.append(f"Vtree0d[{j}]")
        writer.writerow(row)
    print(f"Flow rate Q and Volume V of 0d model will be output to {QV_0ddat}.")
    P_0ddat = output_directory + "P_0d.dat"
    with open(P_0ddat, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = ["time [s]"]
        for i in range(0, 20):
            row.append(f"p[{i}]")
        for i in range(20,56):
            j = int(zd.lookup_value(zd.arterynode,1,2*i,0))
            row.append(f"Ptree0d[{j}]")
        if sim_LMA == 1:
            for i in range(nzm_cowoutlet,nzeromodel,2):
                j = zd.lookup_value(zd.arterynode,1,i,0)
                row.append(f"Ptree0d[{j}]")
        writer.writerow(row)
    print(f"Pressure P of 0d model will be output to {P_0ddat}.")

QV_0dmeancsv = output_directory +"QV_0dmean.csv"
with open(QV_0dmeancsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no"]
    for i in range(0, 20):
        row.append(f"q[{i}]")
        row.append(f"v[{i}]")
    if sim_LMA == 1:
        for i in range(0,numlma):
            row.append(f"Qtree0d[{nartery+1+i}]")
    for i in range(20,56):
        j = zd.lookup_value(zd.arterynode,1,2*i,0)
        row.append(f"Qtree0d[{j}]")
        row.append(f"Vtree0d[{j}]")
    if sim_LMA == 1:
        for i in range(nzm_cowoutlet,nzeromodel,2):
            j = zd.lookup_value(zd.arterynode,1,i,0)
            row.append(f"Qtree0d[{j}]")
            row.append(f"Vtree0d[{j}]")
    writer.writerow(row)
P_0dmeancsv = output_directory + "P_0dmean.csv"
with open(P_0dmeancsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no"]
    for i in range(0, 20):
        row.append(f"p[{i}]")
    for i in range(20,56):
        j = int(zd.lookup_value(zd.arterynode,1,2*i,0))
        row.append(f"Ptree0d[{j}]")
    if sim_LMA == 1:
        for i in range(nzm_cowoutlet,nzeromodel,2):
            j = zd.lookup_value(zd.arterynode,1,i,0)
            row.append(f"Ptree0d[{j}]")
    writer.writerow(row)

# adjusted parameters data
# Adjusted Peripheral resistance (except for CoW outlet)
PR_Adjustedcsv = output_directory + "PR_Adjusted.csv"
with open(PR_Adjustedcsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no.","Relative error","Adjusting coefficient","Total adjusted PR"]
    for i in range(1, nartery+1):
        if mbif_par[i,0] in [0,4] and (i<58 or i>70):
            row.append(f"{i} {artery_name[i,1]} [mmHg*s/ml]")
    writer.writerow(row)
    row = [0, 0, 0, 0]
    for itree in range(1, nartery+1):
        if mbif_par[itree,0] in [0,4] and (itree<58 or itree>70): # exclude cerebral arteries
            row.append(RLCtree[2,itree,1] + RLCtree[2,itree,2])
    writer.writerow(row)
print(f"Adjusted Peripheral resistance PR will be output to {PR_Adjustedcsv}.")
# Adjusted PR for CoW outlet
PR_AdjustedCoWoutletcsv = output_directory + "PR_Adjusted_CoW_outlet.csv"
with open(PR_AdjustedCoWoutletcsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cyclo no.","Relative error"]
    for i in [58,70,61,67,63,65]:
        row.append(f"{i} {artery_name[i,1]} Rtotal[mmHg*s/ml]")
        row.append(f"{i} R1[mmHg*s/ml]")
        row.append(f"{i} R2[mmHg*s/ml]")
        if sim_LMA == 1:
            row.append(f"{i} R3[mmHg*s/ml]")
    if sim_LMA == 1:
        for i in range(0,numlma):
            row.append(f"{nartery+1+i} RLMA[mmHg*s/ml]")
    writer.writerow(row)
    row = [0,0]
    for i in range(0,6):
        arteryno = n_CoW[i] # artery no. of the CoW
        if sim_LMA == 1:
            row.append(RLCtree[2,arteryno,1] + RLCtree[2,arteryno,2] + RLCtree[2,arteryno,3])
        elif sim_LMA == 0:
            row.append(RLCtree[2,arteryno,1] + RLCtree[2,arteryno,2])
        row.append(RLCtree[2,arteryno,1])
        row.append(RLCtree[2,arteryno,2])
        if sim_LMA == 1:
            row.append(RLCtree[2,arteryno,3])
    if sim_LMA == 1:
        for i in range(numlma):
            LMAno = 84+i
            row.append(RLCtree[2,LMAno,1])
    writer.writerow(row)
print(f"Adjusted Peripheral resistance PR of CoW outlet will be output to {PR_AdjustedCoWoutletcsv}.")

# log
# convergence test
log_convergenceCSV = output_directory + "log_convergence.csv"
with open(log_convergenceCSV, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no.", "Q_max_rel_err", "Q_CoW_proximal_max_rel_err",
                     "Parm_rel_err"]
    if sim_LMA == 1:
        row.append("Q_CoW_distal_max_rel_err")
        row.append("LMA_rel_err")
    row.append("Converge_err")
    row.append("if_conv")
    writer.writerow(row)
print(f"Convergence log will be output to {log_convergenceCSV}.")

# elapsed time
if simplified_output == 0:
    time_logdat = output_directory + "time_log.dat"
    with open(time_logdat, mode='w', newline='', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = ["time in sim cycle [s]", "t_total [s]", "t_1d_lw [s]","t_stn [s]",
                "t_0d [s]", "t_1d_bif [s]", "t_reg [s]", "t_out [s]"]
        writer.writerow(row)
    print(f"Elapsed time log will be output to {time_logdat}.")

time_logcyclecsv = output_directory + "time_log_cycle.csv"
with open(time_logcyclecsv, mode='w', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)
    row = ["cycle no.", "t_total [s]", "t_1d_lw [s]","t_stn [s]",
            "t_0d [s]", "t_1d_bif [s]", "t_reg [s]", "t_out [s]"]
    writer.writerow(row)
print(f"Elapsed time log per cycle will be output to {time_logcyclecsv}.")

# visualization
# Create output directory for visualization
import os
if visualization != 0:
    output_directory_viz = output_directory + "visualization/"
    if not os.path.exists(output_directory_viz):
        os.makedirs(output_directory_viz)
        print(f"Output directory '{output_directory_viz}' created.")
    else:
        print(f"Output directory '{output_directory_viz}' already exists.")
    Q_1dvizdat = output_directory_viz + "Q_1d_viz.dat"
    P_1dvizdat = output_directory_viz + "P_1d_viz.dat"
    A_1dvizdat = output_directory_viz + "A_1d_viz.dat"
    print(f"Files for visualization will be output to {output_directory_viz}.")

# new initial condition
# Create output directory for new initial conditions
output_directory_newinitial = output_directory + "new_initial_conditions/"
if not os.path.exists(output_directory_newinitial):
    os.makedirs(output_directory_newinitial)
    print(f"Output directory '{output_directory_newinitial}' created.")
else:
    print(f"Output directory '{output_directory_newinitial}' already exists.")
OneDIC_QAPdat = output_directory_newinitial + "1DIC_QAP_new.dat"
ZeroDICdat = output_directory_newinitial + "0DIC_new.dat"
PPICcsv = output_directory_newinitial + "PPIC_new.csv"

# Array and time step Initialization

# calculating number of time steps
nduration = int(Tduration / dt) # total time step in each cardiac cycle (Tduration: cardiac cycle duration [s])
nlast = nofduration * nduration # total time step in the simulation (nofduration: number of cardiac cycles to compute)
print(f"dt = {dt} [s]\nnduration: no. of cardiac cycles to compute = {nduration}\nnlast: total time step in the simulation = {nlast}")

# calculating cardiac time according to Tduration
tee = 0.3 * math.sqrt(Tduration) # Moment ventricle contractility reaches the peak
tac = 0.68 * (Tduration / 0.855) # Moment atrium begins to contract
tar = 0.83 * (Tduration / 0.855) # Moment atrium begins to relax
tcr = 0.0 # Current time in the cardiac cycle (initialization)

# Initialization of 1d variables (Q,A,U,P)
# 1d array
Qtree = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # flow rate of each artery [ml/s]
Qtreem = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # Q array for t+dt [ml/s]
Atreem = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # A array for t+dt [m^2]
Atreem1 = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # A array for t+dt/2 [m^2]
Utree = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # U array for t [m/s]
Utreem = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # U array for t+dt [m/s]
Utreem1 = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # U array for t+dt/2 [m/s]
Ptree = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # pressure of each artery [Pa]
Ptreem = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C') # P array for t+dt [Pa]
P1u = np.zeros((nartery+1), dtype=np.float64) # P array for 0d calculated values [Pa]
Qguess = np.zeros(nartery+1, dtype=np.float64)
# 1d middle param (j+1/2)
inv_A0mid = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C')
tbetamid = np.zeros((nartery+1, imax_83+1), dtype=np.float64, order='C')
inv_A0mid,tbetamid = lw.parammid_cal(nartery,imaxtree,A0,tbeta,inv_A0mid,tbetamid,cow_geo,exclude_artery)
# 0d array
Vtree0d = np.zeros((nartery+numlma+numcowoutlet+1), dtype=np.float64)
Qtree0d = np.zeros((nartery+numlma+numcowoutlet+1), dtype=np.float64) # terminal arteries and lumped LMA
Ptree0d = np.zeros((nartery+numlma+numcowoutlet+1), dtype=np.float64)
v = np.zeros(20, dtype=np.float64) 
dv = np.zeros(20, dtype=np.float64)
q = np.zeros(20, dtype=np.float64)
dvq = np.zeros(nzeromodel, dtype=np.float64)
P_0d = np.zeros(20, dtype=np.float64)
result = np.zeros((nzeromodel, 2), dtype=np.float64) # initialize the result array for 0D model 0 = cur, 1 = next
# for 0d computation
Dif_ratio = np.zeros(nartery + 1, dtype=np.float64) # initialize the difference ratio array
Q_temp = np.zeros(nartery + 1, dtype=np.float64) # initialize the temporary Q array
A_temp = np.zeros(nartery + 1, dtype=np.float64) # initialize the temporary A array
W1 = np.zeros(nartery+1, dtype=np.float64)
W2 = np.zeros(nartery+1, dtype=np.float64)
A_term1 = np.zeros(nartery+1, dtype=np.float64) # initialize the term1 of A
remuda0 = np.zeros(nartery+1, dtype=np.float64) # initialize the remuda0 array
rukuk = np.zeros((nzeromodel,4),dtype=np.float64) # array to save derivatives for each order
arterynode_map = np.full(nzeromodel, -1, dtype=np.int64)
i_upper_or_lower = np.full(nzm_lma, -1, dtype=np.int64)
for i in range(40, nzm_lma, 2):
    idx0 = zd.lookup_value(zd.arterynode, 1, i, 0)
    if cow_geo != 0 and mbif_par[idx0,0] == 5: 
        idx0 = mbif_par[idx0,1]
    arterynode_map[i] = idx0
    arterynode_map[i+1] = idx0
    idx2 = zd.lookup_value(zd.arterynode, 1, i, 2)
    i_upper_or_lower[i] = idx2
if sim_LMA == 1:
    for i in range(nzm_cowoutlet, nzeromodel,2):
        idx0 = zd.lookup_value(zd.arterynode, 1, i, 0)
        arterynode_map[i] = idx0
        arterynode_map[i+1] = idx0
artery_to_0dindex = np.full(nartery + numcowoutlet + 1, -1, dtype=np.int64)
for i in range(1, nartery + numcowoutlet + 1):
    idx = zd.lookup_value(zd.arterynode, 0, i, 1)
    artery_to_0dindex[i] = idx

# [*,*,1] is the new value (*treem), [*,*,0] is the old value (*tree)
C_jac = np.zeros((3,3), dtype=np.float64) 

# Initialize the stenosis array
dQ_stn = np.zeros(4,dtype = np.float64)  

# array initialization for convergence test
Qoutsum_CoW = np.zeros((nartery+numlma+1,3), dtype=np.float64) # for pr_reg_cereb/LMA 0:R1(4Dflow), 1:R2(SPECT), 2:RLMA(LMA)
Qoutmean_CoW = np.zeros((nartery+numlma+1,3), dtype=np.float64) # for pr_reg_cereb/LMA 0:R1(4Dflow), 1:R2(SPECT), 2:RLMA(LMA)
Qoutmean_CoW_err = np.zeros((nartery+numlma+1,3), dtype=np.float64) # for pr_reg_cereb/LMA; error array 0:R1(4Dflow), 1:R2(SPECT), 2:RLMA(LMA)
Qinmean_CoW = np.zeros((nartery+1,nofduration+1), dtype=np.float64) # for r_com_reg
Rtotalchange = 1.0

# array for mean output and convergence test
Asumtree = np.zeros(nartery+1, dtype=np.float64)
Ameantree = np.zeros((nartery+1,nofduration+1), dtype=np.float64)
Psumtree = np.zeros(nartery+1, dtype=np.float64)
Pmeantree = np.zeros((nartery+1,nofduration+1), dtype=np.float64)
Qsumtree = np.zeros(nartery+1, dtype=np.float64)
Qmeantree = np.zeros((nartery+1,nofduration+1), dtype=np.float64)
Qmeantree_err = np.zeros((nartery + 1), dtype=np.float64) # initialize the error array
QV0dsum = np.zeros(nzeromodel, dtype=np.float64)
QV0dmean = np.zeros((nzeromodel,nofduration+1), dtype=np.float64)
P0dsum = np.zeros(20, dtype=np.float64)
P0dmean = np.zeros((20,nofduration+1), dtype=np.float64)
Ptree0dsum = np.zeros(nartery+numlma+numcowoutlet+1, dtype=np.float64)
Ptree0dmean = np.zeros((nartery+numlma+numcowoutlet+1,nofduration+1), dtype=np.float64)

# array and function for LMA calculation
# # LMA parameters

# import Thomas LMA geometry from CSV
numLMAavg = np.zeros(numlma, dtype=np.float64)
radLMAavg = np.zeros(numlma, dtype=np.float64)
with open('input_csv/LMAgeometry.csv', mode='r', newline='', encoding = 'utf-8') as file:
    reader = csv.reader(file)
    next(reader) # skip header
    for i,row in enumerate(reader):
        numLMAavg[i] = float(row[1])
        radLMAavg[i] = float(row[2])

print("LMA geometry imported from LMAgeometry.csv:")
for i in range(numlma):
    LMAno = 84 + i
    print(f"LMA artery no.{LMAno}: LMA count avg = {numLMAavg[i]}, LMA radius avg = {radLMAavg[i]} mm")

# calculate initial RLMA and set to RLCtree if RLMA_calc == 1
if RLMA_calc == 1:
    for i in range(numlma):
        LMAno = 84 + i
        RLCtree[2,LMAno,1] = pm.RLMAcal(frm, radLMAavg[i], numLMAavg[i])
        print(f"Test Initial RLMA of LMA artery no.{LMAno} RLMA = {RLCtree[2,LMAno,1]} mmHg.s/ml")

disregard_LMA = np.zeros(numlma, dtype=np.int64) # array to indicate whether to disregard LMA artery in optimization (1: disregard, 0: use)
for i in range(numlma):
    if numLMAavg[i] < 1.0 and radLMAavg[i] < 0.1015:
        disregard_LMA[i] = 1
        LMAno = 84 + i
        print(f"Disregarding LMA artery no.{LMAno}. LMA count avg = {numLMAavg[i]}, LMA radius avg = {radLMAavg[i]} mm")

# vsurg parameters
Tot_cyc = 0 # total number of cycles
Tot_cyc_vsurg = 0 # total number of cycles before vsurg (for vsurg=1 only, and this is total cycles before vsurg)

# Cardiac variables
Cval = np.zeros(13, dtype=np.float64) # cardiac variables for 1st category
Eval = np.zeros(6, dtype=np.float64) # cardiac variables for 2nd category
Avalve = np.zeros(4, dtype=np.float64) # valve variables (aortic, mitral, tricuspid, pulmonary)
# set FL,FR1
Cval[0] = 1.0 # FL
Cval[1] = 1.0 # FR1

# Measure computation time
import time
from time import perf_counter
t_1d_lw = 0.0 # time for 1D model Lax-Wendroff calculation
t_stn = 0.0 # time for stenosis calculation
t_0d = 0.0 # time for 0D model calculation
t_1d_bif = 0.0 # time for 1D model bifurcation calculation
t_reg = 0.0 # time for regulation calculation
t_out = 0.0 # time for output and regulation calculation
t_total = 0.0 # total time for the calculation
niter0d = 0 # number of iterations for 0D model

# convergence parameters
conv_all = 0
conv_p17 = 0 if PR_reg_total == 1 else 1
conv_cerebral_4dflow = 0 if PR_reg_cereb == 1 else 1
conv_cerebral_SPECT = 0 if sim_LMA == 1 else 1
conv_LMA = 0 if (sim_LMA == 1 and PR_reg_LMA == 1) else 1
if_conv = 0

# initial condition 
# reference pressure
p0 = p0mmhg * mmhgtoPa # [mmHg] to [Pa]
print(f"Initial reference pressure p0 = {p0} [Pa]")
if init_1d == 0: # if initialize, set to 0 (except for A)
    for itree in range(1, nartery+1):
        if (cow_geo != 0 and exclude_artery[itree] == 1):
            continue
        for j in range(0, imaxtree[itree]+1):
            Atree[itree,j] = A0[itree,j] 
            Atreem[itree,j] = A0[itree,j]
            Qtree[itree,j] = 0.0
            Qtreem[itree,j] = 0.0
            Utree[itree,j] = 0.0
            Utreem[itree,j] = 0.0
else: # initial condition set using the previous data (1DIC_QAP.dat)
    with open("input_csv/1DIC_QAP.dat", mode='r', encoding = 'utf-8') as file:
        reader = csv.reader(file)
        for itree in range(1, nartery+1):
            nmaxtr = imaxtree[itree]
            for j in range(0, nmaxtr+1):
                row = next(reader)
                Qtree[itree,j] = float(row[0])
                Atree[itree,j] = float(row[1])
                Ptree[itree,j] = float(row[2])

    for itree in range(1, nartery+1):
        for j in range(0, imaxtree[itree]+1):
            Atreem[itree,j] = Atree[itree,j]
            Qtreem[itree,j] = Qtree[itree,j]
            Utree[itree,j] = Qtree[itree,j] / Atree[itree,j]
            Utreem[itree,j] = Utree[itree,j]
            Ptreem[itree,j] = Ptree[itree,j]
    p0 = Ptree[1,0] # reference pressure [Pa]

# Load 0D Q,V initial condition from 0DIC.dat
print("Initialize 0d model conditions.")
# Use the same 0DIC.dat file to use literature data for 0D model
# Rename the outputted file: 0DIC_new.dat to 0DIC.dat and set init_0d == 1 to use the previous data 
folder_0dic = "input_csv/" # folder containing 0DIC.dat
file_0dic = folder_0dic + "0DIC.dat"
with open(file_0dic, mode='r', encoding='utf-8') as file:
    lines = file.readlines()
    for i in range(nzeromodel):
        result[i, 0] = np.float64(lines[i].strip())

# set the initial values of Q and V in the 0D model
for i in range(0,20):
    q[i] = result[2*i,0] # Q
    v[i] = result[2*i+1,0] # V

# set the initial values of Q and V of the terminal artery nodes
# if using the previous data, set terminal artery nodes from the data read from 0DIC.dat
if init_0d == 1: # for sim_LMA ==1
    print("Using previous 0D model results for initial condition.")
    for i in range(40,112,2):
        j = zd.lookup_value(zd.arterynode, 1, i, 0)
        Vtree0d[j] = result[i+1,0]
        Qtree0d[j] = result[i,0]
    for i in range(0,numlma):
        Qtree0d[nartery+1+i] = result[i+112,0] # set the LMA flow rate
    for i in range(6):
        idx = 119 + 2 * i
        Qtree0d[arterynode_map[idx]] = result[idx,0]
        Vtree0d[arterynode_map[idx]] = result[idx+1,0]
# else, initialize terminal artery nodes
else: 
    for i in range(40,112,2):
        j = zd.lookup_value(zd.arterynode, 1, i, 0)
        Vtree0d[j] = p0 / mmhgtoPa / RLCtree[3,j,1] # V = p0mmhg * C
        Qtree0d[j] = 0.0 # initialize
        result[i+1,0] = Vtree0d[j]
        result[i,0] = Qtree0d[j]
    for i in range(0,numlma):
        Qtree0d[nartery+1+i] = 0.0 # init the LMA flow rate
        result[i+112,0] = Qtree0d[nartery+1+i] 
    for i in range(6):
        idx = 119 + 2 * i
        idx2 = 40 + 2 * i
        Qtree0d[arterynode_map[idx]] = 0.0
        Vtree0d[arterynode_map[idx]] = p0 / mmhgtoPa / RLCtree[3,arterynode_map[idx2],1]
        result[idx,0] = Qtree0d[arterynode_map[idx]]
        result[idx+1,0] = Vtree0d[arterynode_map[idx]]

# Set result[*,1] to the initial values
for i in range(0,nzeromodel):
    result[i,1] = result[i,0]   

# print if cow_geo != 0
if cow_geo != 0:
    for i in range(58,72):
        print(f"mbif_par of {i}: {mbif_par[i,:]},Initial PR of {i} {artery_name[i,1]}: R1 = {RLCtree[2,i,1]:.2f} [mmHg*s/ml], R2 = {RLCtree[2,i,2]:.2f} [mmHg*s/ml], C = {1.0 / RLCtree[3,i,1]*1.0e3:.2f} [ml/mmHg]")

# DEBUG MODE
# !!! TO RUN THE SIMULATION NORMALLY, SET IS_DEBUG = 0 !!!

# dubug mode (1) or not (0)
is_debug = 0 # 1: debug mode, 0: normal mode
output_debug_results = 1 # output special results for debug?(all cell data for certain no. of cycle) Yes:1, No:0

if is_debug == 1:
    
    print(f"!!!!!DEBUG MODE ENABLED!!!!!")

    # Create output directory for debug outputs
    output_directory_debugoutput = output_directory + "debug_output/"
    if not os.path.exists(output_directory_debugoutput):
        os.makedirs(output_directory_debugoutput)
        print(f"Output directory '{output_directory_debugoutput}' created.")
    else:
        print(f"Output directory '{output_directory_debugoutput}' already exists.")
    
    test_few_loops = 0 # test small num of loops? Yes:1, No:0
    num_of_debug_loop = 10 # number of loops
    
    if output_debug_results == 1:

        print(f"!!!!!DEBUG RESULTS OUTPUT ENABLED!!!!!")

        debug_print_cycle = 1 # print cycle number for debug results
        if debug_print_cycle > num_of_debug_loop:
            debug_print_cycle = num_of_debug_loop
            print(f"debug_print_cycle set to {debug_print_cycle} to match num_of_debug_loop.")

        file_debug_a1d = output_directory_debugoutput + f"DEBUG_A_1d_cycle_{debug_print_cycle}.csv" # A1d file name
        file_debug_p1d = output_directory_debugoutput + f"DEBUG_P_1d_cycle_{debug_print_cycle}.csv" # P1d file name
        file_debug_q1d = output_directory_debugoutput + f"DEBUG_Q_1d_cycle_{debug_print_cycle}.csv" # Q1d file name
        file_debug_aorta = output_directory_debugoutput + f"DEBUG_Q_Aorta.csv" # Aorta file name

        # file initialization
        with open(file_debug_a1d, mode='w', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = ["cell no."]
            for i in range(1, nartery+1):
                row.append(f"{i} {artery_name[i,1]} [ml/s]")
            writer.writerow(row)

        with open(file_debug_p1d, mode='w', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = ["cell no."]
            for i in range(1, nartery+1):
                row.append(f"{i} {artery_name[i,1]} [m/s]")
            writer.writerow(row)

        with open(file_debug_q1d, mode='w', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = ["cell no."]
            for i in range(1, nartery+1):
                row.append(f"{i} {artery_name[i,1]} [mmHg]")
            writer.writerow(row)

        with open(file_debug_aorta, mode='w', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = ["cell no."]
            for i in range(0, imaxtree[1]+1):
                row.append(f"{i}")
            writer.writerow(row)

# Starting calculation loop

# !!!!!-----Start calculation loop-----!!!!!
print("********************Start calculation loop********************")

nbegin = 1
nnn = nbegin # initialize

# for hot start, edited by Thomas 240226
checkpoint_file = "simulation_checkpoint.npz"

if os.path.exists(checkpoint_file):
    print(">>> Found previous run state. Overwriting zeros for hot-start...")
    data = np.load(checkpoint_file)

    # Fill the arrays you just initialized with the saved data
    Qtree[:] = data['Qtree']
    Atree[:] = data['Atree']
    Utree[:] = data['Utree']
    Ptree[:] = data['Ptree']
    Vtree0d[:] = data['Vtree0d']
    Qtree0d[:] = data['Qtree0d']
    result[:] = data['result']
    RLCtree[:] = data['RLCtree']

else:
    print(">>> No previous run state found. Starting with fresh initialization...")

while nnn < nlast + 1: # time step loop

    # for debug
    if is_debug == 1 and test_few_loops == 1:
        print(f"starting cycle no. = {nnn}")
    
    t_str = perf_counter() # obtain time

    # !!!!!-----Step 1: 1D Lax-Wendroff Calculation-----!!!!!

    # print(f"Time step: {nnn}")

    # # Merged Lax-Wendroff calculation
    [Atreem, Ptreem, Utreem, Qtreem] = lw.LaxWendroff(nartery, imaxtree, visc_kr,
                    Atree, Atreem1, A0, inv_A0mid, Utree, Utreem1,
                    tbeta, tbetamid, p0, Atreem, Ptreem, Qtreem, Utreem, dt, dxi, roi, fr,
                    cow_geo, exclude_artery)
    
    # alternative
    # [Atreem, Ptreem, Utreem, Qtreem] = lw.LaxWendroff_opt(nartery, imaxtree, visc_kr, Atree, Atreem1, A0, inv_A0mid, Utree, Utreem1,
    #                 tbeta, tbetamid, p0, Atreem, Ptreem, Qtreem, Utreem, dt, dxi, roi, fr, cow_geo, exclude_artery)

    # measure time for 1D Lax-Wendroff calculation
    t_end = perf_counter() # obtain time
    t_1d_lw = t_1d_lw + t_end - t_str # time for 1D Lax-Wendroff calculation

    # !!!!!-----Step 2: Stenosis Calculation-----!!!!!

    t_str = perf_counter() # obtain time

    if mstn == 1: # if simulate stenosis
        # stenosis calculation
        if nnn == nbegin:
            c_relax_stn = 0.005
            converge_cri_stn = 0.005
            itermax_stn = 30000 # maximum number of iterations for stenosis calculation
        [Atreem, Qtreem, Utreem, Ptreem, Atree, Qtree, Utree] = stn.interface_stn(nnn,numb_stn,nst_ran,
                    byr_st,Atreem,Qtreem,Utreem,Ptreem,Atree,Qtree,Utree,A0,p0,
                    dx,dt,roi,tbeta,c_relax_stn,converge_cri_stn,itermax_stn)

    # measure time for Stenosis calculation
    t_end = perf_counter() # obtain time
    t_stn = t_stn + t_end - t_str # time Stenosis calculation

    # !!!!!-----Step 3: 0-1d Coupling Computation-----!!!!!

    t_str = perf_counter() # obtain time

    # 0-1D Coupling Calculation
    if nnn == nbegin:
        itermax_0d = 300 # maximum number of iterations for 0D model (default: 300)
        iterrelax_0d = 5000 # number of iterations for relaxation factor
        c_relax_01d_coup = 0.1 # relaxation factor for 0-1D coupling
        converge_cri_01d_coup = 0.001 # convergence criterion for 0-1D coupling

    # calculate the current step in the cardiac cycle
    tcr = dt * float(nnn) - Tduration * int(dt * float(nnn) / Tduration)

    # for debug
    # print_cycle = 100
    # if nnn % print_cycle == 0:
    #     print(f"  Time step: {nnn}, tcr: {tcr:.4f} [s]")

    (Atreem, Qtreem, Utreem, Vtree0d, Qtree0d, Qguess, P_0d, 
     q, v, dv, dvq, Ptree0d, niter_0dcount, result, Cval, Eval, Avalve) = zd.interface_01d(nnn,nartery,
                    imaxtree,Atreem,Qtreem,Utreem,Qtree,Atree,
                    A0,P1u,p0,mbif_par,Qguess,nzeromodel,Dif_ratio,Q_temp,A_temp,W1,W2,rukuk,
                    Vtree0d,Ptree0d,Qtree0d,v,dv,q,dvq,P_0d,result,tee,tac,tar,tcr,
                    dx,dt,roi,tbeta,A_term1,remuda0,c_relax_01d_coup,itermax_0d,Tduration,nbegin,
                    iterrelax_0d,converge_cri_01d_coup,mmhgtoPa,RLCtree,numlma,nzm_lma,nzm_cowoutlet,Cval,Eval,Avalve,
                    arterynode_map,i_upper_or_lower,artery_to_0dindex,sim_LMA,disregard_LMA)

    # measure time for 0-1D Coupling Calculation
    t_end = perf_counter() # obtain time
    t_0d = t_0d + t_end - t_str # time for 0-1D Coupling Calculation
    niter0d += niter_0dcount # total number of iterations for 0D model

    # !!!!!-----Step 4: Bifurcation Computation and Pressure Calculation at Arterial ends-----!!!!!

    t_str = perf_counter() # obtain time

    # Bifurcation Calculation
    (Qtreem,Atreem,Utreem,Ptreem,C_jac) = bif.bifurcation(nnn,C_jac,Qtree,Atree,Qtreem,Atreem,Utreem,Ptreem,
                                                          p0,A0,ro,dx,dt,tbeta,mbif_par,imaxtree,nartery,cow_geo,exclude_artery)

    # measure time for Bifurcation Calculation and Pressure Calculation at Arterial ends
    t_end = perf_counter() # obtain time
    t_1d_bif = t_1d_bif + t_end - t_str # time for Bifurcation Calculation

    # !!!!!-----Step 5: Regulate total peripheral resistance to match measured arterial pressure    -----!!!!!
    # !!!!!-----      & cerebral terminal resistace to match measured cerebral blood flow rates     -----!!!!!
    # !!!!!-----      & radii of communicating arteries to match measured cerebral blood flow rates -----!!!!!

    t_str = perf_counter() # obtain time

    kprint = 10 # amount of divisions in 1 cardiac cycle
    if nnn % (nduration/kprint) == 0:
        k = int((nnn/nduration) * kprint) % kprint
        cycle = int(nnn/nduration)+1
        if k == 0:
            k = kprint
            cycle = int(nnn/nduration)
        print(f"{nnn} time step(s) out of {nlast} completed({cycle}th cardiac cycle {k}/{kprint}). Time elapsed = {t_total:.2f} [s].")
    if nnn % nduration == 0: # at every cardiac cycle
        print(f"Finished Cycle no. = {nnn / nduration}. Time elapsed = {t_total:.2f} [s]")

    # calculate sum flow rate through cow outlet artery (every time step)
    for i in range(0, 6): # 6 outlets of CoW
        arteryno = n_CoW[i]
        Qoutsum_CoW[arteryno,0] += Qtreem[arteryno,imaxtree[arteryno]] * 1.0e6 # for R1(4Dflow)
        Qoutsum_CoW[arteryno,1] += Qtree0d[arteryno] # for R2(SPECT)
    for i in range(0,numlma): # 7 LMA
        Qoutsum_CoW[i+nartery+1,2] += result[i+112,1] # for RLMA(LMA)

    # calculate sum flow rate for every artery (at the middle of the artery)
    pm.update_sumtree(Qsumtree, Qtreem, imaxtree, nartery, param = 1.0e6) # Qsumtree
    pm.update_sumtree(Psumtree, Ptreem, imaxtree, nartery, param = 1.0/mmhgtoPa) # Psumtree
    pm.update_sumtree(Asumtree, Atreem, imaxtree, nartery, param = 1.0e4) # Asumtree
    QV0dsum[:] += result[:,1]
    P0dsum[:] += P_0d[:]
    Ptree0dsum[:] += Ptree0d[:]

    # # calculate sum pressure at arm (artery no. 17)
    # Parm_sum += Ptreem[17,int(imaxtree[17]/2)] / mmhgtoPa

    # at the end of the cardiac cycle, calculate the mean and reset the sum
    if nnn % nduration == 0:
        cycle_no = int(nnn / nduration) # current cardiac cycle number

        Qoutmean_CoW[:,:] = Qoutsum_CoW[:,:] / nduration
        Qoutsum_CoW[:,:] = 0.0
        Qmeantree[:,cycle_no] = Qsumtree[:] / nduration
        Qsumtree[:] = 0.0
        Pmeantree[:,cycle_no] = Psumtree[:] / nduration
        Psumtree[:] = 0.0
        Ameantree[:,cycle_no] = Asumtree[:] / nduration
        Asumtree[:] = 0.0

        QV0dmean[:,cycle_no] = QV0dsum[:] / nduration
        QV0dsum[:] = 0.0

        P0dmean[:,cycle_no] = P0dsum[:] / nduration
        Ptree0dmean[:,cycle_no] = Ptree0dsum[:] / nduration
        P0dsum[:] = 0.0
        Ptree0dsum[:] = 0.0

        Qinmean_CoW[40,cycle_no] = Qmeantree[40,cycle_no]
        Qinmean_CoW[47,cycle_no] = Qmeantree[47,cycle_no]
        Qinmean_CoW[56,cycle_no] = Qmeantree[56,cycle_no]

    # every two cardiac cycles, test if converged and regulate
    if nnn % (nduration * 2) == 0:

        cycle_no = int(nnn / nduration) # current cardiac cycle number

        print(f"Checking convergence...")

        # Q convergence test at every artery (if new time step is close to the previous time step)
        # yes -> conv_all = 1, no -> conv_all = 0
        rel_err_Qartery = 0.0
        irelmax = 0
        for i in range(1, nartery + 1):
            if Qmeantree[i,cycle_no] == 0.0:
                Qmeantree_err[i] = 0.0
            else:
                Qmeantree_err[i] = abs( (Qmeantree[i,cycle_no] - Qmeantree[i,cycle_no-1]) / Qmeantree[i,cycle_no] )
            if Qmeantree_err[i] > rel_err_Qartery:
                rel_err_Qartery = Qmeantree_err[i] # find the maximum relative error
                irelmax = i # find the index of the maximum relative error
        # rel_err_Qartery = np.max(Qmeantree_err[1:nartery+1]) # find the maximum relative error
        # irelmax = np.argmax(Qmeantree_err[1:nartery+1]) # find the index of the maximum relative error
        if rel_err_Qartery < converge_err:
            conv_all = 1
            print(f"Q converged at every artery! Max rel_err_Qartery = {rel_err_Qartery} at artery no. {irelmax}")
        else:
            conv_all = 0
            print(f"Q not converged at every artery! Max rel_err_Qartery = {rel_err_Qartery} at artery no. {irelmax}")

        # P convergence test at artery no. 17 (if p17 is close to parm_ref)
        # yes -> conv_p17 = 1, no -> conv_p17 = 0
        if PR_reg_total == 1:
            Parm_mean = Pmeantree[17,cycle_no] # mean pressure at artery no. 17 [mmHg]
            rel_err_Parm = abs( (Parm_mean - Parm_ref) / Parm_ref )
            if rel_err_Parm < converge_err_Parm:
                conv_p17 = 1
                # PR_reg_total = 0 # set PR_reg_total to 0 if converged
                print(f"Parm(artery no. 17) converged! rel_err_Parm = {rel_err_Parm}")
            else:
                conv_p17 = 0
                print(f"Parm(artery no. 17) not converged! rel_err_Parm = {rel_err_Parm}")
        else:
            conv_p17 = 1
            rel_err_Parm = 0.0

        # Q convergence test at cerebral peripheral arteries (if flow rate is close to measured flow rate)
        # yes -> conv_cerebral = 1, no -> conv_cerebral = 0
        if PR_reg_cereb == 1:
            rel_err_QCoWOut = 0.0
            irelmax = 0
            for i in range(0, 6): # calculate the rel err for R1 (4Dflow)
                arteryno = n_CoW[i] # artery no. of the CoW
                Qoutmean_CoW_err[arteryno,0] = abs( (Qoutmean_CoW[arteryno,0] - Qref[arteryno,0]) / Qref[arteryno,0] )
                if Qoutmean_CoW_err[arteryno,0] > rel_err_QCoWOut:
                    rel_err_QCoWOut = Qoutmean_CoW_err[arteryno,0]
                    irelmax = arteryno
            
            if rel_err_QCoWOut < converge_err_cereb: # converge_err for default
                conv_cerebral_4dflow = 1
                # PR_reg_cereb = 0 # set PR_reg_cereb to 0 if converged
                print(f"Qoutmean_CoW converged! Max rel_err_QCoWOut = {rel_err_QCoWOut} at artery no. {irelmax}")
            else:
                conv_cerebral_4dflow = 0
                print(f"Qoutmean_CoW not converged! Max rel_err_QCoWOut = {rel_err_QCoWOut} at artery no. {irelmax}")
        else:
            conv_cerebral_4dflow = 1
            rel_err_QCoWOut = 0.0
            
        if sim_LMA == 1 and PR_reg_cereb == 1:      

            rel_err_LMA = 0.0     
            irelmax = 0
            for i in range(0,numlma): # calculate the rel err for RLMA (LMA)
                Qoutmean_CoW_err[nartery+1+i,2] = abs((abs(Qoutmean_CoW[nartery+1+i,2]) - abs(Qref[84+i,2])) / abs(Qref[84+i,2]))
                if disregard_LMA[i] == 1:
                    continue
                if Qoutmean_CoW_err[nartery+1+i,2] > rel_err_LMA:
                    rel_err_LMA = Qoutmean_CoW_err[nartery+1+i,2]
                    irelmax = nartery+1+i

            rel_err_SPECT = 0.0
            irelmaxSPECT = 0
            for i in range(0,6): # calculate the rel err for R2 (SPECT)
                arteryno = n_CoW[i]
                Qoutmean_CoW_err[arteryno,1] = abs( (Qoutmean_CoW[arteryno,1] - Qref[arteryno,1]) / Qref[arteryno,1] )
                if Qoutmean_CoW_err[arteryno,1] > rel_err_SPECT:
                    rel_err_SPECT = Qoutmean_CoW_err[arteryno,1]
                    irelmaxSPECT = arteryno

            if rel_err_SPECT < converge_err_cereb: # converge_err for default
                conv_cerebral_SPECT = 1
                print(f"Qoutmean_CoW (SPECT) converged! Max rel_err_SPECT = {rel_err_SPECT} at artery no. {irelmaxSPECT}")
            else:
                conv_cerebral_SPECT = 0
                print(f"Qoutmean_CoW (SPECT) not converged! Max rel_err_SPECT = {rel_err_SPECT} at artery no. {irelmaxSPECT}")


            if PR_reg_LMA == 1:
                if rel_err_LMA < converge_err_LMA: # converge_err for default
                    conv_LMA = 1
                    print(f"Qoutmean_LMA converged! Max rel_err_LMA = {rel_err_LMA} at LMA no. {irelmax}")
                else:
                    conv_LMA = 0
                    print(f"Qoutmean_LMA not converged! Max rel_err_LMA = {rel_err_LMA} at LMA no. {irelmax}")

            else:
                conv_LMA = 1
                rel_err_LMA = 0.0

            # print out the flow rates of LMA and CoW outlets
            for i in range(0,numlma):
                print(f"LMA no. {84+i}: Qref = {Qref[84+i,2]:.2f} [ml/s], Qoutmean_CoW = {Qoutmean_CoW[nartery+1+i,2]:.2f} [ml/s], rel_err_LMA = {Qoutmean_CoW_err[nartery+1+i,2]:.4f}")
            for i in range(0,6):
                arteryno = n_CoW[i]
                print(f"CoW outlet (proximal;4DFlow) artery no. {arteryno}: Qref = {Qref[arteryno,0]:.2f} [ml/s], Qoutmean_CoW = {Qoutmean_CoW[arteryno,0]:.2f} [ml/s], rel_err_CoW = {Qoutmean_CoW_err[arteryno,0]:.4f}")
                print(f"CoW outlet (distal;SPECT) artery no. {arteryno}: Qref = {Qref[arteryno,1]:.2f} [ml/s], Qoutmean_CoW = {Qoutmean_CoW[arteryno,1]:.2f} [ml/s], rel_err_CoW = {Qoutmean_CoW_err[arteryno,1]:.4f}")

        else:
            conv_LMA = 1
            conv_cerebral_SPECT = 1
            rel_err_LMA = 0.0
            rel_err_SPECT = 0.0

        # if all converged, set if_conv = 1, else if_conv = 0
        if conv_all == 1 and conv_p17 == 1 and conv_cerebral_4dflow == 1 and conv_cerebral_SPECT == 1 and conv_LMA == 1:
            if_conv = 1
            print(f"All converged! if_conv = {if_conv}.")
        else:
            if_conv = 0
            print(f"Everything hasn't converged! if_conv = {if_conv}.")

        # if if_conv == 0, regulate if any regulations are active
        if if_conv == 0:

            # change total peripheral resistance to fit mean blood pressure if PR_reg_total == 1
            if PR_reg_total == 1:
                Parm_mean = Pmeantree[17,cycle_no] # mean pressure at artery no. 17 [mmHg]
                coeff_pr = 1.0 - alpha * (Parm_mean - Parm_ref) / Parm_ref # coefficient to change peripheral resistance
                for i in range(1, nartery + 1):
                    if mbif_par[i,0] == 0 and (i<58 or i>70): # exclude cerebral arteries
                        R_total = RLCtree[2,i,1] + RLCtree[2,i,2] # total peripheral resistance
                        RLCtree[2,i,2] -= R_total * alpha * (Parm_mean - Parm_ref) / Parm_ref 
                        # RLCtree[2,i,1] *= coeff_pr # change peripheral resistance
                        # RLCtree[2,i,2] *= coeff_pr # change peripheral resistance
                        if RLCtree[2,i,2] < 0.0:
                            RLCtree[2,i,2] = 0.0
                            print(f"RLCtree[2,{i},2] < 0.0, set to 0.0")
                Rtotalchange *= coeff_pr # update the total peripheral resistance 

            # change cerebral peripheral resistance to fit measured blood flow rate if PR_reg_cerebral == 1 
            if PR_reg_cereb == 1: # regulating cerebral peripheral resistance

                if sim_LMA == 0: # if not simulating LMA
                    for i in range(0,6):
                        arteryno = n_CoW[i] # artery no. of the CoW
                        coeff_4DFLOW = 1.0 - alpha * (Qref[arteryno,0] - Qoutmean_CoW[arteryno,0]) / Qref[arteryno,0]
                        coeff_CoW[arteryno,0] = coeff_4DFLOW # store the coefficient for CoW regulation

                        RLCtree[2,arteryno,1] *= coeff_CoW[arteryno,0] # change CoW peripheral resistance
                        RLCtree[2,arteryno,2] *= coeff_CoW[arteryno,0]

                if sim_LMA == 1: # if simulating LMA

                    # regulating R1-R3: 4Dflow and SPECT
                    for i in range(0,6):
                        arteryno = n_CoW[i] # artery no. of the CoW
                        coeff_CoW[arteryno,0] = 1.0 - alpha_4DFlow * (Qref[arteryno,0] - Qoutmean_CoW[arteryno,0]) / Qref[arteryno,0] # store the coefficient for CoW regulation
                        coeff_CoW[arteryno,1] = 1.0 - alpha_SPECT * (Qref[arteryno,1] - Qoutmean_CoW[arteryno,1]) / Qref[arteryno,1] # store the coefficient for SPECT regulation
                        
                        # change R1,R2 for 4Dflow regulation
                        RLCtree[2,arteryno,1] *= coeff_CoW[arteryno,0] # change CoW peripheral resistance R1
                        RLCtree[2,arteryno,2] *= coeff_CoW[arteryno,0]

                        # change R3 for SPECT regulation
                        RLCtree[2,arteryno,3] *= coeff_CoW[arteryno,1]

                        print(f"R1,R2(4Dflow),R3(SPECT) of {arteryno} updated to {RLCtree[2,arteryno,1]:.4f}, {RLCtree[2,arteryno,2]:.4f}, {RLCtree[2,arteryno,3]:.4f} for regulation")

                    if PR_reg_LMA == 1:

                        print("Regulating LMA resistance...")
                        # change RLMA <- If main trunk converged
                        if conv_cerebral_4dflow == 1 and conv_cerebral_SPECT == 1:
                            for i in range(0,numlma):
                                if Qref[84+i,2] * Qoutmean_CoW[nartery+1+i,2] > 0.0:
                                    coeff_CoW[84+i,2] = 1.0 - alpha_LMA * (abs(Qref[84+i,2]) - abs(Qoutmean_CoW[nartery+1+i,2])) / abs(Qref[84+i,2]) # store the coefficient for LMA regulation
                                    RLCtree[2,84+i,1] *= coeff_CoW[84+i,2] # change LMA peripheral resistance RLMA
                                    print(f"RLMA of {84+i}: coeff_CoW = {coeff_CoW[84+i,2]}, RLMA = {RLCtree[2,84+i,1]} for regulation")

            # change radii of communicating arteries to fit measured flow rates if r_com_reg == 1
            # !!! currently, we do not know what this is meant to fit !!! 
            if r_com_reg == 1:
                cycle_no = int(nnn / nduration) # current number of cardiac cycles
                for j in [64,59,69]: # ACom (No.64), Rt.PCom (No.59), Lt.PCom (No.69)
                    if j == 64:
                        Q_reg = Qinmean_CoW[40,cycle_no] + Qinmean_CoW[47,cycle_no]
                        Qref_reg = Qref[40,0] + Qref[47,0]
                        rfac = (Qref_reg * Qinmean_CoW[40,cycle_no] 
                                - Q_reg * Qref[40,0]) / (Q_reg * Qref[40,0])
                    if j == 59:
                        Q_reg = Qinmean_CoW[47,cycle_no] + Qinmean_CoW[56,cycle_no]
                        Qref_reg = Qref[47,0] + Qref[56,0]
                        rfac = (Qref_reg * Qinmean_CoW[47,cycle_no] 
                                - Q_reg * Qref[47,0]) / (Q_reg * Qref[47,0])
                    if j == 69:
                        Q_reg = Qinmean_CoW[40,cycle_no] + Qinmean_CoW[56,cycle_no]
                        Qref_reg = Qref[40,0] + Qref[56,0]
                        rfac = (Qref_reg * Qinmean_CoW[40,cycle_no] 
                                - Q_reg * Qref[40,0]) / (Q_reg * Qref[40,0])
                    if Qinmean_CoW[j,cycle_no] < 0.0:
                        rfac = - 1.0 * rfac
                    for i in range(0,imaxtree[j]+1):
                        Rtree0[j,i] = Rtree0[j,i] * (1.0 + 2.0 * rfac) # original coeff: 0.5
                        Atree[j,i]  = pi * (Rtree0[j,i]**2.0)
                        A0[j,i]     = Atree[j,i]
                        Atreem[j,i] = Atree[j,i]
                        Ptreem[j,i] = Ptreem[j,0]
                        Utreem[j,i] = Qtreem[j,0] / Atreem[j,i]
                        
        # if if_conv == 1 do one last loop of calculation
        elif if_conv == 1:
            print(f"Convergence criteria has been met. Performing calculation of last cardiac cycle.")
            nlast = nnn # set the last time step to be the current cardiac cycle

    # save the previous time step Q,A,P,U results for next step calculation
    Qtree[:,:] = Qtreem[:,:]
    Atree[:,:] = Atreem[:,:]
    Utree[:,:] = Utreem[:,:]
    Ptree[:,:] = Ptreem[:,:]

    # measure time for Regulation Calculation
    t_end = perf_counter() # obtain time
    t_reg = t_reg + t_end - t_str # time for Regulation Calculation

    # !!!!!-----Step 6: Output Results-----!!!!!

    t_str = perf_counter() # obtain time

    # output time step results (A,P,Q,U,QV,P_0d)
    # At first step
    if nnn == nbegin:
        time = 0.0
    # At every time step
    if nnn > 0 and nnn % nprint == 0:
        time = nnn * dt
    # output
    if nnn == nbegin or (nnn > 0 and nnn % nprint == 0):

        # unit conversion (1d)
        for itree in range(1, nartery+1):
            imax = imaxtree[itree]+1
            Atree_output[itree,:imax] = Atree[itree,:imax] * 1.0e4 # convert m^2 to cm^2
            Qtree_output[itree,:imax] = Qtree[itree,:imax] * 1.0e6 # convert 
            Ptree_output[itree,:imax] = Ptree[itree,:imax] / mmhgtoPa # convert Pa to mmHg
            Utree_output[itree,:imax] = Qtree_output[itree,:imax] / Atree_output[itree,:imax] 

        # Atree_output[:,:] = Atree[:,:] * 1.0e4
        # Qtree_output[:,:] = Qtree[:,:] * 1.0e6
        # Ptree_output[:,:] = Ptree[:,:] / mmhgtoPa
        # Utree_output[:,:] = np.where(Atree_output != 0.0, Qtree_output / Atree_output, 0.0)

        # output 
        if simplified_output == 0:
            with open(Q_1ddat, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{time:.4f}"]
                for itree in range(1, nartery+1):
                    row.append(Qtree_output[itree,int(imaxtree[itree]/2)])
                writer.writerow(row)
            with open(U_1ddat, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{time:.4f}"]
                for itree in range(1, nartery+1):
                    row.append(Utree_output[itree,int(imaxtree[itree]/2)])
                writer.writerow(row)
            with open(P_1ddat, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{time:.4f}"]
                for itree in range(1, nartery+1):
                    row.append(Ptree_output[itree,int(imaxtree[itree]/2)])
                writer.writerow(row)
            with open(A_1ddat, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{time:.4f}"]
                for itree in range(1, nartery+1):
                    row.append(Atree_output[itree,int(imaxtree[itree]/2)])
                writer.writerow(row)

            with open(QV_0ddat, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{time:.4f}"]
                for i in range(0,20):
                    row.append(result[2*i,1])
                    row.append(result[2*i+1,1])
                if sim_LMA == 1:
                    for i in range(112,119):
                        row.append(result[i,1])
                for i in range(20,56):
                    row.append(result[2*i,1])
                    row.append(result[2*i+1,1])
                if sim_LMA == 1:
                    for i in range(119,131,2):
                        row.append(result[i,1])
                        row.append(result[i+1,1])
                writer.writerow(row)
            with open(P_0ddat, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{time:.4f}"]
                for i in range(0,20):
                    row.append(P_0d[i])
                for i in range(20,56):
                    j = int(zd.lookup_value(zd.arterynode,1,2*i,0))
                    row.append(Ptree0d[j])
                if sim_LMA == 1:
                    for i in range(112,119):
                        j = int(zd.lookup_value(zd.arterynode,1,i,0))
                        row.append(Ptree0d[j])
                writer.writerow(row)

    # output new initial conditions
    # 1DIC_QAP_new.dat, ZeroDIC_new.dat, PPIC_new.csv
    if nnn == nlast:
        with open(OneDIC_QAPdat, mode='w', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            for itree in range(1, nartery+1):
                for j in range(0, imaxtree[itree]+1):
                    row = [Qtree[itree,j], Atree[itree,j], Ptree[itree,j]]
                    writer.writerow(row)
        with open(ZeroDICdat, mode='w', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            for i in range(0,nzeromodel):
                row = [result[i,1]]
                writer.writerow(row)
        with open(PPICcsv, mode='w', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = ["artery no.", "R1[mmHg.s/ml]", "R2[mmHg.s/ml]", "R3[mmHg.s/ml]", "C[ml/mmHg]", "L[mmHg.s2/ml]"]
            writer.writerow(row)
            for i in range(1,nartery+1): # artery no., R1, R2, R3 (RLCtree(2,j,1-3)), C(RLCtree(3,j,1)), L(RLCtree(1,j,2))
                if mbif_par[i,0] in [0,4]:
                    row = [i, RLCtree[2,i,1], RLCtree[2,i,2], RLCtree[2,i,3], 1.0 / RLCtree[3,i,1], 1.0 / RLCtree[1,i,2]]
                    writer.writerow(row)
            for i in range(0,numlma):
                LMAno = 84 + i
                row = [LMAno, RLCtree[2,LMAno,1], RLCtree[2,LMAno,2], RLCtree[2,LMAno,3], 1.0 / RLCtree[3,LMAno,1], 1.0 / RLCtree[1,LMAno,2]]
                writer.writerow(row)

        # for hot start, edited by Thomas 240226
        np.savez(checkpoint_file,
                 Qtree=Qtree, Atree=Atree, Utree=Utree, Ptree=Ptree,
                 Vtree0d=Vtree0d, Qtree0d=Qtree0d, result=result, RLCtree=RLCtree)
        print(f">>> Hot-start state saved to {checkpoint_file}")
    
    # output visualization data
    if ((visualization == 1 and (nnn >= viz_str * nduration and nnn <= viz_end * nduration)) or (visualization == 2 and (nnn >= nlast-nduration and nnn <= nlast))):

        time = nnn * dt

        # ファイルをバイナリ書き込み＆ランダムアクセスモードで開く
        f_q = open(Q_1dvizdat, "r+b") if os.path.exists(Q_1dvizdat) else open(Q_1dvizdat, "w+b")
        f_p = open(P_1dvizdat, "r+b") if os.path.exists(P_1dvizdat) else open(P_1dvizdat, "w+b")
        f_a = open(A_1dvizdat, "r+b") if os.path.exists(A_1dvizdat) else open(A_1dvizdat, "w+b")

        recl = 8  # 1レコード = 8バイト
        count_rec = 1  # Fortranと同様に1から開始

        if nnn >= viz_str * nduration and nnn <= viz_end * nduration and nnn % nprint == 0:

            # timeを書き込み（float64）
            pm.write_record(f_q, count_rec, time, recl)
            pm.write_record(f_p, count_rec, time, recl)
            pm.write_record(f_a, count_rec, time, recl)
            count_rec += 1

            for j in range(1,nartery+1):
                ncell = int(Dtree[j] * dxi)
                # ncell 書き込み（int → float64にするか、あるいは明示的にint32として書くか）
                pm.write_record(f_q, count_rec, ncell, recl, fmt='i')
                pm.write_record(f_p, count_rec, ncell, recl, fmt='i')
                pm.write_record(f_a, count_rec, ncell, recl, fmt='i')
                count_rec += 1

                for i in range(0,imaxtree[j]+1):
                    pm.write_record(f_q, count_rec, Qtree_output[j][i], recl)  # float64
                    pm.write_record(f_p, count_rec, Ptree_output[j][i], recl)
                    pm.write_record(f_a, count_rec, Atree_output[j][i], recl)
                    count_rec += 1

        # ファイルを閉じる
        f_q.close()
        f_p.close()
        f_a.close()

    # output convergence data
    if nnn % (nduration * 2) == 0:
        cycle_no = int(nnn / nduration) # current number of cardiac cycles
        with open(PR_Adjustedcsv, mode='a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [cycle_no, rel_err_Parm, coeff_pr, Rtotalchange]
            for itree in range(1, nartery+1):
                if mbif_par[itree,0] in [0,4] and (itree<58 or itree>70): # exclude cerebral arteries
                    row.append(RLCtree[2,itree,1] + RLCtree[2,itree,2])
            writer.writerow(row)
        with open(PR_AdjustedCoWoutletcsv, mode='a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [cycle_no, rel_err_QCoWOut]
            for i in range(0,6):
                arteryno = n_CoW[i] # artery no. of the CoW
                row.append(RLCtree[2,arteryno,1] + RLCtree[2,arteryno,2])
                row.append(RLCtree[2,arteryno,1])
                row.append(RLCtree[2,arteryno,2])
                if sim_LMA == 1:
                    row.append(RLCtree[2,arteryno,3])
            if sim_LMA == 1:
                for i in range(numlma):
                    LMAno = 84+i
                    row.append(RLCtree[2,LMAno,1])
            writer.writerow(row)
        with open(log_convergenceCSV, mode='a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [cycle_no, rel_err_Qartery, rel_err_QCoWOut,
                     rel_err_Parm]
            if sim_LMA == 1:
                row.append(rel_err_SPECT)
                row.append(rel_err_LMA)
            row.append(converge_err)
            row.append(if_conv)
            writer.writerow(row)

    # !!!DEBUG OUTPUT!!!
    if is_debug == 1 and output_debug_results == 1:
        
        if nnn == debug_print_cycle:
            with open(file_debug_a1d, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                for j in range(0,imax_83+1):
                    row = [f"{j}"]
                    for i in range(1, nartery+1):
                        row.append(f"{Atree_output[i,j]:.6f}")
                    writer.writerow(row)

            with open(file_debug_p1d, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                for j in range(0,imax_83+1):
                    row = [f"{j}"]
                    for i in range(1, nartery+1):
                        row.append(f"{Ptree_output[i,j]:.6f}")
                    writer.writerow(row)

            with open(file_debug_q1d, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                for j in range(0,imax_83+1):
                    row = [f"{j}"]
                    for i in range(1, nartery+1):
                        row.append(f"{Qtree_output[i,j]:.6f}")
                    writer.writerow(row)

        if nnn % nprint ==0:
            with open(file_debug_aorta, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{nnn/nduration}"]
                for i in range(0,imaxtree[1]+1):
                    row.append(f"{Qtree_output[1,i]:.16f}")
                writer.writerow(row)

    # output cycle mean data
    if nnn % nduration == 0 and nnn > 0:

        cycle = int(nnn / nduration) # current number of cardiac cycles
        cycleidx = cycle
        if Tot_cyc_vsurg != 0:
            cycleidx -= Tot_cyc_vsurg

        # output A, P, Q, QV0d cycle mean
        with open(Q_1dmeancsv, mode = 'a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [f"{cycleidx}"]
            for itree in range(1, nartery+1):
                row.append(Qmeantree[itree,cycle]) # convert
            writer.writerow(row)
        with open(P_1dmeancsv, mode = 'a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [f"{cycleidx}"]
            for itree in range(1, nartery+1):
                row.append(Pmeantree[itree,cycle])
            writer.writerow(row)
        with open(A_1dmeancsv, mode = 'a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [f"{cycleidx}"]
            for itree in range(1, nartery+1):
                row.append(Ameantree[itree,cycle]) # convert
            writer.writerow(row)  
        with open(QV_0dmeancsv, mode = 'a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [f"{cycleidx}"]
            for i in range(0,20):
                row.append(QV0dmean[2*i,cycle])
                row.append(QV0dmean[2*i+1,cycle])
            if sim_LMA == 1:
                for i in range(112,119):
                    row.append(QV0dmean[i,cycle])
            for i in range(20,56):
                row.append(QV0dmean[2*i,cycle])
                row.append(QV0dmean[2*i+1,cycle])
            if sim_LMA == 1:
                for i in range(119,131,2):
                    row.append(QV0dmean[i,cycle])
                    row.append(QV0dmean[i+1,cycle])
            writer.writerow(row)
        with open(P_0dmeancsv, mode = 'a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [f"{cycleidx}"]
            for i in range(0,20):
                row.append(P0dmean[i,cycle])
            for i in range(20,56):
                j = int(zd.lookup_value(zd.arterynode,1,2*i,0))
                row.append(Ptree0dmean[j,cycle])
            if sim_LMA == 1:
                for i in range(nzm_cowoutlet,nzeromodel,2):
                    j = zd.lookup_value(zd.arterynode,1,i,0)
                    row.append(Ptree0dmean[j,cycle])
            writer.writerow(row)    

    # measure time for Output Results
    t_end = perf_counter() # obtain time
    t_out = t_out + t_end - t_str # time for Output Results

    # total time for every time step
    t_total = t_1d_lw + t_stn + t_0d + t_1d_bif + t_reg + t_out

    # output time_log
    if simplified_output == 0:
        if nnn == nbegin or (nnn > 0 and nnn % nprint == 0):
            with open(time_logdat, mode = 'a', newline='', encoding = 'utf-8') as file:
                writer = csv.writer(file)
                row = [f"{time:.4f}", t_total, t_1d_lw, t_stn, 
                    t_0d, t_1d_bif, t_reg, t_out]
                writer.writerow(row)
    
    if nnn % nduration == 0 and nnn > 0:
        cycle = int(nnn / nduration) # current number of cardiac cycles
        with open(time_logcyclecsv, mode = 'a', newline='', encoding = 'utf-8') as file:
            writer = csv.writer(file)
            row = [f"{cycle}", t_total, t_1d_lw, 
                t_stn, t_0d, t_1d_bif, 
                t_reg, t_out]
            writer.writerow(row)

    # !!!!!-----End calculation loop-----!!!!!

    # DEBUG MODE            
    if is_debug == 1:
        
        if test_few_loops == 1:
            
            if nnn == num_of_debug_loop:
                
                print(f"Debug Loop completed!")
                print(f"Total {num_of_debug_loop} loop(s) completed.")
                print(f"Finishing calculation.")
                stop_flag = 1
                # sys.exit(0)

    # if last time step, switch the flag to finish the calculation
    if nnn == nlast:
        print(f"Last time step reached! Finishing the calculation.")
        stop_flag = 1
        print(f"stop_flag set to {stop_flag}.")

    # if stop_flag == 1, finish the calculation
    if stop_flag == 1:

        Tot_cyc = int(nnn / nduration)
        print(f"Total number of cycles = {Tot_cyc}")

        # for debug
        if is_debug == 1:
            break

        if if_conv == 1:
            print("All convergence criteria has been met.")
        else:
            print("All convergence criteria has not been met, but calculation is finished due to max number of iteration being done.")

        # if no vsurg_stn, finish the calculation
        if vsurg_stn == 0: 
            print(f"******************** End calculation loop ********************")
            break

        # if vsurg_stn == 1, remove the stenosis and continue the calculation
        elif vsurg_stn != 0:
            if mstn == 0:
                print(f"Stenosis calculation has not been done! Finish the calculation.")
                print(f"If this is unexpected, check if mstn = 1 in the input file.")
                print(f"******************** End calculation loop ********************")
                break
            elif mstn == 1: # remove stenosis and continue the calculation

                print(f"Performing virtual surgery (removing stenosis)...")
                
                if vsurg_stn == 1: # single-stenosis removal or double-stenosis removal(both at the same time)
                    # Remove stenosis shape from 1D tree
                    # exp for only stenosed area
                    for n in range(1, numb_stn+1):
                        j_stn = nst_ran[n,0]
                        nst_str = nst_ran[n,1]
                        nst_end = nst_ran[n,2]
                        nst_length = nst_end - nst_str
                        dR = Rtree0[j_stn,nst_end] - Rtree0[j_stn,nst_str]
                        for i in range(nst_str,nst_end+1):
                            Rtree0[j_stn,i] = Rtree0[j_stn,nst_str] * exp(log(Rtree0[j_stn,nst_end] / Rtree0[j_stn,nst_str]) * (float(i) - float(nst_str)) / float(nst_length))
                            # Rtree0[j_stn,i] = Rtree0[j_stn,nst_str] + float(i - nst_str) / float(nst_length) * dR # alternative: linear
                            A0[j_stn,i] = pi * (Rtree0[j_stn,i]**2.0)
                            Atree[j_stn,i] = A0[j_stn,i]
                        # for j in range(0,imaxtree[j_stn]+1): # print
                            # print(f"Artery {j_stn}, node {j}, Rtree0: {Rtree0[j_stn,j]:.6f} mm, Atree: {Atree[j_stn,j]:.6f} mm^2")
                        print(f"Stenosis removed from artery {j_stn}, nodes {nst_str} to {nst_end}.")
                    # another way: setting stenosed artery to exponential shape between R0 and R1(geometry == 1)
                    # for n in range(1, numb_stn+1):
                    #     i = nst_ran[n,0]
                    #     for j in range(0, imaxtree[i]+1):
                    #         Rtree0[i,j] = Rtree[i,0] * exp(log(Rtree[i,1] / Rtree[i,0]) * (dx * float(j)) / Dtree[i] )
                    #         Atree[i,j] = pi * (Rtree0[i,j] ** 2.0)
                    #         A0[i,j] = Atree[i,j]
                    #         print(f"Artery {i}, node {j}, Rtree0: {Rtree0[i,j]:.6f} mm, Atree: {Atree[i,j]:.6f} mm^2")

                elif vsurg_stn in [2,3]: # multiple-stenosis removal (one by one)
                    if vsurg_stn == 2:
                        n = 1 # j_stn = 40 Lt. ICA
                        j_stn = nst_ran[n,0]
                        if j_stn != 40: # switch if not Lt. ICA
                            n = 2
                    elif vsurg_stn == 3:
                        n = 2 # j_stn = 47 Rt. ICA
                        j_stn = nst_ran[n,0]
                        if j_stn != 47: # switch if not Rt. ICA
                            n = 1
                    # Remove stenosis shape from 1D tree
                    j_stn = nst_ran[n,0]
                    nst_str = nst_ran[n,1]
                    nst_end = nst_ran[n,2]
                    nst_length = nst_end - nst_str
                    dR = Rtree0[j_stn,nst_end] - Rtree0[j_stn,nst_str]
                    for i in range(nst_str,nst_end+1):
                        Rtree0[j_stn,i] = Rtree0[j_stn,nst_str] * exp(log(Rtree0[j_stn,nst_end] / Rtree0[j_stn,nst_str]) * (float(i) - float(nst_str)) / float(nst_length))
                        # Rtree0[j_stn,i] = Rtree0[j_stn,nst_str] + float(i - nst_str) / float(nst_length) * dR # alternative: linear
                        A0[j_stn,i] = pi * (Rtree0[j_stn,i]**2.0)
                        Atree[j_stn,i] = A0[j_stn,i]
                    # for j in range(0,imaxtree[j_stn]+1): # print
                        # print(f"Artery {j_stn}, node {j}, Rtree0: {Rtree0[j_stn,j]:.6f} mm, Atree: {Atree[j_stn,j]:.6f} mm^2")
                    print(f"Stenosis removed from artery {j_stn}, nodes {nst_str} to {nst_end}.")

                # re-calculate tbeta after removing stenosis
                tbeta, cof_ela, c_ela_aor = tb.tbetacal(tbeta, cof_ela, Rtree0, imaxtree, nartery, age, roi, cow_geo, exclude_artery, c_ela_aor)
                inv_A0mid,tbetamid = lw.parammid_cal(nartery,imaxtree,A0,tbeta,inv_A0mid,tbetamid,cow_geo,exclude_artery)

                # testing: set 1d and 0d values to initial condition after removing stenosis
                # set the initial values of A, Q, P in the 1D model
                if vsurg_stn_qap_init == 1:
                    print(f"Setting qap to 0 for 1D and 0D models after virtual surgery...")
                    for itree in range(1, nartery+1):
                        if (cow_geo != 0 and exclude_artery[itree] == 1):
                            continue
                        for j in range(0, imaxtree[itree]+1):
                            Atree[itree,j] = A0[itree,j] 
                            Atreem[itree,j] = A0[itree,j]
                            Qtree[itree,j] = 0.0
                            Qtreem[itree,j] = 0.0
                            Utree[itree,j] = 0.0
                            Utreem[itree,j] = 0.0
                            Ptree[itree,j] = 0.0
                            Ptreem[itree,j] = 0.0
                    folder_0dic = "input_csv/" # folder containing 0DIC.dat
                    file_0dic = folder_0dic + "0DIC.dat"
                    with open(file_0dic, mode='r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for i in range(nzeromodel):
                            result[i, 0] = np.float64(lines[i].strip())
                    # set the initial values of Q and V in the 0D model
                    # for i in range(0,20): # for q,v 0-19, do nothing?
                    #     q[i] = result[2*i,0] # Q
                    #     v[i] = result[2*i+1,0] # V
                    for i in range(40,112,2):
                        j = zd.lookup_value(zd.arterynode, 1, i, 0)
                        Vtree0d[j] = p0 / mmhgtoPa / RLCtree[3,j,1] # V = p0mmhg * C
                        Qtree0d[j] = 0.0 # initialize
                        result[i+1,0] = Vtree0d[j]
                        result[i,0] = Qtree0d[j]
                    for i in range(0,numlma):
                        Qtree0d[nartery+1+i] = 0.0 # init the LMA flow rate
                        result[i+112,0] = Qtree0d[nartery+1+i] 
                    for i in range(6):
                        idx = 119 + 2 * i
                        idx2 = 40 + 2 * i
                        Qtree0d[arterynode_map[idx]] = 0.0
                        Vtree0d[arterynode_map[idx]] = p0 / mmhgtoPa / RLCtree[3,arterynode_map[idx2],1]
                        result[idx,0] = Qtree0d[arterynode_map[idx]]
                        result[idx+1,0] = Vtree0d[arterynode_map[idx]]
                elif vsurg_stn_qap_init == 0:
                    print(f"Keeping previous time step values for qap of 1D and 0D models after virtual surgery...")

                # Switch mstn flag:
                if vsurg_stn == 1:
                    vsurg_stn = 0
                    mstn = 0 # stop simulating stenosis
                elif vsurg_stn in [2,3]:
                    # modify stenosis parameters
                    numb_stn -= 1
                    if vsurg_stn == 2:
                        n = 1 # j_stn = 40 Lt. ICA
                        j_stn = nst_ran[n,0]
                        if j_stn != 40: # switch if not Lt. ICA
                            n = 2
                    elif vsurg_stn == 3:
                        n = 2 # j_stn = 47 Rt. ICA
                        j_stn = nst_ran[n,0]
                        if j_stn != 47: # switch if not Rt. ICA
                            n = 1
                    if n == 1: # delete nst_ran[1,:] and move nst_ran[2,:] to nst_ran[1,:]
                        nst_ran[1,0] = nst_ran[2,0]
                        nst_ran[1,1] = nst_ran[2,1]
                        nst_ran[1,2] = nst_ran[2,2]
                        nst_ran[1,3] = nst_ran[2,3]
                    # if n == 2, do nothing
                    # set vsurg_stn to 1 to continue 
                    vsurg_stn = 1
                    mstn = 1 # continue simulating stenosis (for multiple stenosis removal)

                    print(f"Continuing stenosis simulation for remaining stenosis(es). Number of remaining stenosis: {numb_stn}.")
                    print(f"Remaining stenosis is at artery no.{nst_ran[1,0]}, from node {nst_ran[1,1]} to node {nst_ran[1,2]}, neck at node {nst_ran[1,3]}.")

                # Switch flags:
                PR_reg_total = 0
                PR_reg_cereb = 0
                PR_reg_LMA = 0
                r_com_reg = 0
                stop_flag = 0 # reset stop_flag to continue calculation
                if_conv = 0 # reset if_conv
                Tot_cyc_vsurg = int(nnn / nduration) # store total cycles before virtual surgery
                print(f"Cycles before virtual surgery: {Tot_cyc_vsurg}.")
                nlast += nofduration * nduration # set the last time step to be additonal double cardiac cycles
                print(f"Stenosis removed! Redoing the calculation.")

    # time step increment

    # for debug
    if is_debug == 1 and test_few_loops == 1:
        print(f"finished step no.{nnn}")

    nnn += 1 # increment the time step

# close log file
sys.stdout.flush() # flush stdout before closing the log file
sys.stdout.close() # close stdout
# log_file.close()

# END OF CODE