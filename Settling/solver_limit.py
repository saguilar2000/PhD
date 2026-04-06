"""
solver for -iw * x = M * x
"""

import numpy as np
from scipy import linalg

def compute_drift(h0, r0, z, Am, n, betay, betaz, chi):
    sqrt_eta = h0 / r0
    f = 1 + chi
    fx = -(sqrt_eta/Am) * ((2+n)/betay + n/betaz) * f
    fy = (sqrt_eta/Am) * ((z/h0) * (r0/h0) * (1/np.sqrt(betay*betaz))) * f
    fz = (sqrt_eta/Am) * ((z/h0) * (r0/h0) * (1/betay)) * f
    
    return fx, fy, fz

def compute_eigenvalues(Kx, Kz, Am, chi, fx, fy, fz, betay, betaz, mn, mi, q, r0, h0, vec, comp_n = 1, comp_i = 1):
    """_summary_

    Args:
        Kx (_type_): k_x eta r0
        Kz (_type_): k_z eta r0
        Am (_type_): Ambipolar diffusion Elsasser number
        chi (_type_): _description_
        fx (_type_): _description_
        fy (_type_): _description_
        fz (_type_): _description_
        betay (_type_): _description_
        betaz (_type_): _description_
        mn (_type_): _description_
        mi (_type_): _description_
        q (_type_): _description_
        r0 (_type_): _description_
        h0 (_type_): _description_
        vec (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Enumeration
    vnx, vny, vnz = 0, 1, 2
    vix, viy, viz = 3, 4, 5
    B_x, B_y, B_z = 6, 7, 8
    p_n, p_i = 9, 10
    # Constants
    K2 = Kx**2 + Kz**2
    csi_csn = mn / mi
    eps = chi * (mi / mn)
    invsqrt_eta = r0 / h0
    f =  1 + chi

    # i\vec{k}\cdot \vec{f_j}
    iKf = Kz * invsqrt_eta * fz
    
    # Magnetic field components
    By0 = np.sqrt(betaz / (betay + betaz))
    Bz0 = np.sqrt(betay / (betay + betaz))
    B0y = np.sqrt(1 + betay / betaz)
    B0z = np.sqrt(1 + betaz / betay)

    # Matrix initialization: A*v = w*B*v
    A = np.zeros([11, 11], dtype='complex128')
    B = np.eye(11, dtype='complex128')

    # Row 3 * eps: v_ix
    A[vix,vix] = -(iKf * eps + Am)                              # prev: -upsilon = -(iKf + Am/eps)
    A[vix,B_x] = 1j * Kz * invsqrt_eta * (2 * f / betaz) * B0z      # prev: 1j * Kz * invsqrt_eta * (2*B0z / (eps*betaz))
    
    B[vix,vix] = eps  # fix

    # Row 4 * eps: v_iy
    A[viy,viy] = -(iKf * eps + Am)                              # prev: -upsilon
    A[viy,B_y] = 1j * Kz * invsqrt_eta * (2 * f / betaz) * B0z      # prev: 1j * Kz * invsqrt_eta * (2*B0z / (eps*betaz))
    
    B[viy,viy] = eps  # fix

    # Row 5 * eps: v_iz
    A[viz,viz] = -(iKf * eps + Am)                              # prev: -upsilon
    A[viz,B_y] = -1j * Kz * invsqrt_eta * (2 * f / betay) * B0y     # prev: -1j * (Kz * r0/h0) * (2*B0y / (eps*betay))
    A[viz,p_i] = (-1j * Kz * invsqrt_eta * (csi_csn) * eps) * comp_i       # prev: -1j * (csi_csn) * (Kz * r0/h0)
    
    B[viz,viz] = eps  # fix

    # Row 6: B_x
    A[B_x,vix] = 1j * Kz * invsqrt_eta * Bz0
    A[B_x,B_x] = -iKf

    # Row 7: B_y
    A[B_y,viy] = 1j * Kz * invsqrt_eta * Bz0
    A[B_y,viz] = -1j * Kz * invsqrt_eta * By0
    A[B_y,B_y] = -iKf

    # Row 8: B_z
    A[B_z,B_z] = 0

    # Row 10: rho_i
    A[p_i,viz] = (-1j * Kz * invsqrt_eta) * comp_i
    A[p_i,p_i] = (-iKf) * comp_i

    v = 0
    if vec == True:
        w, v = linalg.eig(A, b=B)
    else:
        w = linalg.eigvals(A, b=B)

    return w, v
