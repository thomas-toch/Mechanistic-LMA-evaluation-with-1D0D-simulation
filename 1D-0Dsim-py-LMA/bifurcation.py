# function for bifurcation calculation
import numpy as np
import math
from numba import njit, prange

sqrt = math.sqrt
pow = math.pow

nriteration = 6 # maximum number of iterations for newton's method (default:12)

@njit(fastmath=True)
def bifurcation(nnn,C_jac,Qtree,Atree,Qtreem,Atreem,Utreem,Ptreem,p0,A0,ro,dx,dt,tbeta,
                mbif_par,imaxtree,nartery,cow_geo,exclude_artery):
    
    if nnn == 1:
        print("Initializing bifurcation calculation.")

    for itree in range(1,nartery+1):
        
        if cow_geo != 0 and exclude_artery[itree] == 1:  # skip excluded arteries
            continue

        C_jac.fill(0.0)  # Initialize the Jacobian matrix

        if mbif_par[itree,0] == 1: #branching
            npn = itree
            nd1 = mbif_par[itree,1]
            nd2 = mbif_par[itree,2]

            (Qtreem,Atreem,Utreem,C_jac) = bifbranch(C_jac,imaxtree,npn,nd1,nd2,Utreem,Qtreem,Atree,Atreem,A0,
                                    ro,dx,dt,tbeta,iteration=nriteration) 

        elif mbif_par[itree,0] == 2: # merging
            npn = mbif_par[itree,2]
            nd1 = itree
            nd2 = mbif_par[itree,1]

            (Qtreem,Atreem,Utreem,C_jac) = bifmerge(C_jac,imaxtree,npn,nd1,nd2,Utreem,Qtreem,Atree,Atreem,A0,
                                   ro,dx,dt,tbeta,iteration=nriteration) 

        else: # no bifurcation
            continue

    # print("Bifurcation calculation completed.")

    # update pressure at arterial ends
    for itree in range(1,nartery+1):
        if cow_geo != 0 and exclude_artery[itree] == 1:
            continue
        j = 0
        Ptreem[itree,j] = tbeta[itree,j] * (sqrt(Atreem[itree,j] / A0[itree,j]) - 1.0) + p0
        j = imaxtree[itree]
        Ptreem[itree,j] = tbeta[itree,j] * (sqrt(Atreem[itree,j] / A0[itree,j]) - 1.0) + p0

    # print("pressure at arterial ends calculated.")

    return (Qtreem,Atreem,Utreem,Ptreem,C_jac)

@njit(fastmath=True)
def bifbranch(C_jac,imaxtree,npn,nd1,nd2,Utreem,Qtreem,Atree,Atreem,A0,ro,dx,dt,tbeta,iteration):

    # print("bifbranch called")

    dx_dt = dx / dt * 0.5
    Aq = - Atree[npn,imaxtree[npn]-1] - Atree[npn,imaxtree[npn]] + Atreem[npn,imaxtree[npn]-1]
    Aq1 = Atreem[nd1,1] - Atree[nd1,0] - Atree[nd1,1]
    Aq2 = Atreem[nd2,1] - Atree[nd2,0] - Atree[nd2,1]
    Cq = Aq + Aq1 + Aq2 + ( - Qtreem[npn,imaxtree[npn]-1] 
                           + Qtreem[nd1,1] + Qtreem[nd2,1] ) / dx_dt
    
    Cp_1 = tbeta[npn,imaxtree[npn]]
    Cp1_1 = tbeta[nd1,0]
    Cp2_1 = tbeta[nd2,0]
    
    Cp_2 = - dx_dt * Aq + Qtreem[npn,imaxtree[npn]-1]
    Cp1_2 = dx_dt * Aq1 + Qtreem[nd1,1]
    Cp2_2 = dx_dt * Aq2 + Qtreem[nd2,1]
    
    Cp_c1 = 0.5 * ro * (Cp_2 * Cp_2) # Mother tube 0.5*ro*q(2) 
    Cp_c2 = - dx_dt * ro * Cp_2           
    Cp_c3 = 0.5 * ro * (dx_dt * dx_dt)
    
    Cp1_c1 = 0.5 * ro * (Cp1_2 * Cp1_2) # Daughter tube 1 
    Cp1_c2 = dx_dt * ro * Cp1_2 
    Cp1_c3 = 0.5 * ro * (dx_dt * dx_dt)
    
    Cp2_c1 = 0.5 * ro * (Cp2_2 * Cp2_2) # Daughter tube 2 
    Cp2_c2 = dx_dt * ro * Cp2_2 
    Cp2_c3 = 0.5 * ro * (dx_dt * dx_dt)
    
    Amu = Atreem[npn,imaxtree[npn]]
    A01u = Atreem[nd1,0]
    A02u = Atreem[nd2,0]

    Amu0 = A0[npn,imaxtree[npn]]
    A01u0 = A0[nd1,0]
    A02u0 = A0[nd2,0]
    sqrtAmu0 = sqrt(Amu0)
    sqrtA01u0 = sqrt(A01u0)
    sqrtA02u0 = sqrt(A02u0)

    for _ in range(iteration):

        C_jac[0, :] = 1.0
        C_jac[1, 0] = (-2.0 * Cp_c1 / (Amu*Amu*Amu) - Cp_c2 / (Amu*Amu) + Cp_1 * 0.5 / sqrtAmu0 / sqrt(Amu))
        C_jac[1, 1] = (2.0 * Cp1_c1 / (A01u*A01u*A01u) + Cp1_c2 / (A01u*A01u) - Cp1_1 * 0.5 / sqrtA01u0 / sqrt(A01u))
        C_jac[1, 2] = 0.0
        C_jac[2, 0] = C_jac[1, 0]
        C_jac[2, 1] = 0.0
        C_jac[2, 2] = (2.0 * Cp2_c1 / (A02u*A02u*A02u) + Cp2_c2 / (A02u*A02u) - Cp2_1 * 0.5 / sqrtA02u0 / sqrt(A02u))

        C_inv = inv3x3(C_jac)

        F1 = Amu + A01u + A02u + Cq
        F2 = (Cp_1 * (sqrt(Amu / Amu0) - 1.0)
              + Cp_c1 / (Amu*Amu) + Cp_c2 / Amu + Cp_c3
              - Cp1_1 * (sqrt(A01u / A01u0) - 1.0)
              - Cp1_c1 / (A01u*A01u) - Cp1_c2 / A01u - Cp1_c3)
        F3 = (Cp_1 * (sqrt(Amu / Amu0) - 1.0)
              + Cp_c1 / (Amu*Amu) + Cp_c2 / Amu + Cp_c3
              - Cp2_1 * (sqrt(A02u / A02u0) - 1.0)
              - Cp2_c1 / (A02u*A02u) - Cp2_c2 / A02u - Cp2_c3)

        Amu -= C_inv[0, 0] * F1 + C_inv[0, 1] * F2 + C_inv[0, 2] * F3
        A01u -= C_inv[1, 0] * F1 + C_inv[1, 1] * F2 + C_inv[1, 2] * F3
        A02u -= C_inv[2, 0] * F1 + C_inv[2, 1] * F2 + C_inv[2, 2] * F3

    # Update the bifurcation flow and area arrays
    Qtreem[npn,imaxtree[npn]] = - dx_dt * (Aq + Amu) + Qtreem[npn,imaxtree[npn] - 1]
    Qtreem[nd1,0] = dx_dt * (Aq1 + A01u) + Qtreem[nd1,1]
    Qtreem[nd2,0] = dx_dt * (Aq2 + A02u) + Qtreem[nd2,1]
    Atreem[npn,imaxtree[npn]] = Amu
    Atreem[nd1,0] = A01u
    Atreem[nd2,0] = A02u
    Utreem[npn,imaxtree[npn]] = Qtreem[npn,imaxtree[npn]] / Atreem[npn,imaxtree[npn]]
    Utreem[nd1,0] = Qtreem[nd1,0] / Atreem[nd1,0]
    Utreem[nd2,0] = Qtreem[nd2,0] / Atreem[nd2,0]

    return (Qtreem,Atreem,Utreem,C_jac)

@njit(fastmath=True)
def bifmerge(C_jac,imaxtree,npn,nd1,nd2,Utreem,Qtreem,Atree,Atreem,A0,ro,dx,dt,tbeta,iteration):

    # print("bifmerge called")

    dx_dt = dx / dt * 0.5
    Aq = - Atree[npn,0] - Atree[npn,1] + Atreem[npn,1]
    Aq1 = - Atree[nd1,imaxtree[nd1]-1] - Atree[nd1,imaxtree[nd1]] + Atreem[nd1,imaxtree[nd1]-1]
    Aq2 = - Atree[nd2,imaxtree[nd2]-1] - Atree[nd2,imaxtree[nd2]] + Atreem[nd2,imaxtree[nd2]-1]
    Cq = Aq + Aq1 + Aq2 + (Qtreem[npn,1] - Qtreem[nd1,imaxtree[nd1]-1] 
                           - Qtreem[nd2,imaxtree[nd2]-1] ) / dx_dt

    Cp_1 = tbeta[npn,0]
    Cp1_1 = tbeta[nd1,imaxtree[nd1]]
    Cp2_1 = tbeta[nd2,imaxtree[nd2]]
    
    Cp_2 = - dx_dt * Aq + Qtreem[npn,1]
    Cp1_2 = dx_dt * Aq1 + Qtreem[nd1,imaxtree[nd1]-1]
    Cp2_2 = dx_dt * Aq2 + Qtreem[nd2,imaxtree[nd2]-1]
    
    Cp_c1 = 0.5 * ro * (Cp_2 * Cp_2) # Mother tube 0.5*ro*q(2)
    Cp_c2 = - dx_dt * ro * Cp_2           
    Cp_c3 = 0.5 * ro * (dx_dt * dx_dt)
    
    Cp1_c1 = 0.5 * ro * (Cp1_2 * Cp1_2) # Daughter tube 1
    Cp1_c2 = dx_dt * ro * Cp1_2
    Cp1_c3 = 0.5 * ro * (dx_dt * dx_dt)
    
    Cp2_c1 = 0.5 * ro * (Cp2_2 * Cp2_2) # Daughter tube 2
    Cp2_c2 = dx_dt * ro * Cp2_2
    Cp2_c3 = 0.5 * ro * (dx_dt * dx_dt)
    
    Amu = Atreem[npn,0]
    A01u = Atreem[nd1,imaxtree[nd1]]
    A02u = Atreem[nd2,imaxtree[nd2]]

    A0mu  = A0[npn,0]
    A01mu = A0[nd1,imaxtree[nd1]]
    A02mu = A0[nd2,imaxtree[nd2]]
    sqrtA0mu  = sqrt(A0mu)
    sqrtA01u0 = sqrt(A01mu)
    sqrtA02u0 = sqrt(A02mu)

    for _ in range(iteration):

        sqrtAmu   = sqrt(Amu)
        sqrtA01u  = sqrt(A01u)
        sqrtA02u  = sqrt(A02u)

        # Jacobian Matrix
        C_jac[0, :] = 1.0
        C_jac[1, 0] = (-2.0 * Cp_c1 / (Amu*Amu*Amu) - Cp_c2 / (Amu*Amu) + Cp_1 * 0.5 / sqrtA0mu / sqrtAmu)
        C_jac[1, 1] = (2.0 * Cp1_c1 / (A01u*A01u*A01u) + Cp1_c2 / (A01u*A01u) - Cp1_1 * 0.5 / sqrtA01u0 / sqrtA01u)
        C_jac[1, 2] = 0.0
        C_jac[2, 0] = C_jac[1, 0]
        C_jac[2, 1] = 0.0
        C_jac[2, 2] = (2.0 * Cp2_c1 / (A02u*A02u*A02u) + Cp2_c2 / (A02u*A02u) - Cp2_1 * 0.5 / sqrtA02u0 / sqrtA02u)

        C_inv = inv3x3(C_jac)

        # Residuals
        F1 = Amu + A01u + A02u + Cq
        F2 = (Cp_1 * (sqrtAmu / sqrtA0mu - 1.0) + Cp_c1 / (Amu*Amu) + Cp_c2 / Amu + Cp_c3
              - Cp1_1 * (sqrtA01u / sqrtA01u0 - 1.0) - Cp1_c1 / (A01u*A01u) - Cp1_c2 / A01u - Cp1_c3)
        F3 = (Cp_1 * (sqrtAmu / sqrtA0mu - 1.0) + Cp_c1 / (Amu*Amu) + Cp_c2 / Amu + Cp_c3
              - Cp2_1 * (sqrtA02u / sqrtA02u0 - 1.0) - Cp2_c1 / (A02u*A02u) - Cp2_c2 / A02u - Cp2_c3)

        # Update
        Amu  -= C_inv[0,0] * F1 + C_inv[0,1] * F2 + C_inv[0,2] * F3
        A01u -= C_inv[1,0] * F1 + C_inv[1,1] * F2 + C_inv[1,2] * F3
        A02u -= C_inv[2,0] * F1 + C_inv[2,1] * F2 + C_inv[2,2] * F3

    # Update the bifurcation flow and area arrays
    Qtreem[npn,0] = dx_dt * (Aq + Amu) + Qtreem[npn,1]
    Qtreem[nd1,imaxtree[nd1]] = - dx_dt * (Aq1 + A01u) + Qtreem[nd1,imaxtree[nd1]-1]
    Qtreem[nd2,imaxtree[nd2]] = - dx_dt * (Aq2 + A02u) + Qtreem[nd2,imaxtree[nd2]-1]
    Atreem[npn,0] = Amu
    Atreem[nd1,imaxtree[nd1]] = A01u
    Atreem[nd2,imaxtree[nd2]] = A02u
    Utreem[npn,0] = Qtreem[npn,0] / Atreem[npn,0]
    Utreem[nd1,imaxtree[nd1]] = Qtreem[nd1,imaxtree[nd1]] / Atreem[nd1,imaxtree[nd1]]
    Utreem[nd2,imaxtree[nd2]] = Qtreem[nd2,imaxtree[nd2]] / Atreem[nd2,imaxtree[nd2]]

    return (Qtreem,Atreem,Utreem,C_jac)

@njit(fastmath=True)
def inv3x3(M):

    # calculate the inverse of a 3x3 matrix M
    det = (
        M[0, 0]*(M[1, 1]*M[2, 2] - M[1, 2]*M[2, 1])
      - M[0, 1]*(M[1, 0]*M[2, 2] - M[1, 2]*M[2, 0])
      + M[0, 2]*(M[1, 0]*M[2, 1] - M[1, 1]*M[2, 0])
    )

    if np.abs(det) < 1e-12:
        return np.eye(3)  # fallback: identity
    
    inv = np.zeros((3, 3), dtype=np.float64)
    inv[0, 0] = (M[1, 1]*M[2, 2] - M[1, 2]*M[2, 1]) / det
    inv[0, 1] = (M[0, 2]*M[2, 1] - M[0, 1]*M[2, 2]) / det
    inv[0, 2] = (M[0, 1]*M[1, 2] - M[0, 2]*M[1, 1]) / det
    inv[1, 0] = (M[1, 2]*M[2, 0] - M[1, 0]*M[2, 2]) / det
    inv[1, 1] = (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0]) / det
    inv[1, 2] = (M[0, 2]*M[1, 0] - M[0, 0]*M[1, 2]) / det
    inv[2, 0] = (M[1, 0]*M[2, 1] - M[1, 1]*M[2, 0]) / det
    inv[2, 1] = (M[0, 1]*M[2, 0] - M[0, 0]*M[2, 1]) / det
    inv[2, 2] = (M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]) / det

    return inv

# fortran code functions
def Migs(A): # changes A inside the function, may lead to a bug
    """
    Compute inverse of A using elgs (Gaussian elimination) with pivoting.
    Returns X such that X @ A = I and A @ X = I.

    A: square numpy array.
    Returns X: inverse matrix.
    """
    n = A.shape[0] # number of rows and columns
    X = np.zeros((n,n), dtype=float) # inverse matrix
    B = np.eye(n) # identity matrix
    indx = np.zeros(n, dtype=int) # index array

    indx = Elgs(A) # perform Gaussian elimination with partial pivoting

    # forward elimination on B
    for i in range(n-1):
        for j in range(i+1, n):
            B[indx[j], :] -= A[indx[j], i] * B[indx[i], :]

    # back substitution
    # last row
    X[n-1, :] = B[indx[n-1], :] / A[indx[n-1], n-1]
    # remaining
    for j in range(n-2, -1, -1):
        X[j, :] = B[indx[j], :].copy()
        for k in range(j+1, n):
            X[j, :] -= A[indx[j], k] * X[k, :]
        X[j, :] /= A[indx[j], j]
    return X


def Elgs(A): 
    """
    Perform partial-pivoting Gaussian elimination on A in-place.
    Returns the pivot index list.

    A: square numpy array, will be modified to contain LU factors.
    Returns: indx, a list of row index permutations (0-based).
    """
    n = A.shape[0] # number of rows and columns
    indx = np.zeros(n, dtype=int) # index array
    for i in range(0,n): # initialize index
        indx[i] = i

    # Find the rescaling factors, one from each row
    C = np.zeros(n, dtype=float) # column scaling
    for i in range(0,n):
        C[i] = np.max(abs(A[i,:])) # maximum absolute value of each row

    # Search the pivoting (largest) element from each column
    for j in range(0,n):
        pivtemp = 0.0
        for i in range(j,n):
            piv = abs(A[indx[i],j]) / C[indx[i]]
            if piv > pivtemp:
                pivtemp = piv
                k = i
        # Interchange rows via indx to reccord pivoting order
        indx[j], indx[k] = indx[k], indx[j]
        for i in range(j+1,n):
            Pj = A[indx[i],j] / A[indx[j],j]
            # record pivoting ratios below the diagonal
            A[indx[i],j] = Pj
            A[indx[i],j+1:n] -= Pj * A[indx[j],j+1:n]

    return indx
