"""
solve: -iw * x = M * x
"""

import numpy as np
from scipy import linalg
import random

MATRIX_ORDER = 6

# ============================================================
# Drift computation
# ============================================================
def compute_drift(h0, r0, z, Am, n, betay, betaz, chi):
    sqrt_eta = h0 / r0
    f = 1 + chi
    fx = -(sqrt_eta/Am) * ((2+n)/betay + n/betaz) * f
    fy = (sqrt_eta/Am) * ((z/h0) * (r0/h0) * (1/np.sqrt(betay*betaz))) * f
    fz = (sqrt_eta/Am) * ((z/h0) * (r0/h0) * (1/betay)) * f
    
    return fx, fy, fz

# ============================================================
# Elsasser number ambipolar diffusion term
# ============================================================
def Am(r, W, z, eps):
    r0 = 1.0 # AU
    h0 = 0.05 * r
    rho_n0 = 1.0e-10
    gamma = 3.5e13  # cm^3 g^-1 s^-1
    G = 6.67430e-8  # cm^3 g^-1 s^-2
    M_sun = 1.989e33  # g
    r_cm = r * 1.496e13  # Convert AU to cm

    rho_n = rho_n0 * (r/r0)**(W) * np.exp(-z**2/(2*h0**2))
    Omega_K = np.sqrt(G*M_sun/r_cm**3)
    alpha = gamma * rho_n / Omega_K

    return alpha * eps

# ============================================================
# Eigenvalue computation
# ============================================================
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
    vix, viy, viz = 0, 1, 2
    B_x, B_y, p_i = 3, 4, 5
    # Constants
    K2 = Kx**2 + Kz**2
    csi_csn = mn / mi
    eps = chi * (mi / mn)
    invsqrt_eta = r0 / h0
    f =  1 + chi

    # i\vec{k}\cdot \vec{f_j}
    iKf = 1j * Kz * invsqrt_eta * fz
    
    # Magnetic field components
    By0 = np.sqrt(betaz / (betay + betaz))
    Bz0 = np.sqrt(betay / (betay + betaz))
    B0y = np.sqrt(1 + betay / betaz)
    B0z = np.sqrt(1 + betaz / betay)

    # Matrix initialization: A*v = w*B*v
    A = np.zeros([MATRIX_ORDER, MATRIX_ORDER], dtype='complex128')
    B = np.eye(MATRIX_ORDER, dtype='complex128')

    # Row 1 * eps: v_ix
    A[vix,vix] = -(iKf * eps)
    A[vix,B_x] = 1j * Kz * invsqrt_eta * (2 * f / betaz) * B0z
    
    B[vix,vix] = eps  # fix

    # Row 2 * eps: v_iy
    A[viy,viy] = -(iKf * eps)
    A[viy,B_y] = 1j * Kz * invsqrt_eta * (2 * f / betaz) * B0z
    
    B[viy,viy] = eps  # fix

    # Row 3 * eps: v_iz
    A[viz,viz] = -(iKf * eps)
    A[viz,B_y] = -1j * Kz * invsqrt_eta * (2 * f / betay) * B0y
    A[viz,p_i] = (-1j * Kz * invsqrt_eta * (csi_csn) * eps) * comp_i
    
    B[viz,viz] = eps  # fix

    # Row 4: B_x
    A[B_x,vix] = 1j * Kz * invsqrt_eta * Bz0
    A[B_x,B_x] = -iKf

    # Row 5: B_y
    A[B_y,viy] = 1j * Kz * invsqrt_eta * Bz0
    A[B_y,viz] = -1j * Kz * invsqrt_eta * By0
    A[B_y,B_y] = -iKf

    # Row 6: rho_i
    A[p_i,viz] = (-1j * Kz * invsqrt_eta) * comp_i
    A[p_i,p_i] = (-iKf) * comp_i

    v = 0
    if vec == True:
        w, v = linalg.eig(A, b=B)
    else:
        w = linalg.eigvals(A, b=B)

    return w, v

def quote():
    try:
        with open("quotes.txt", "r") as f:
            lines = f.readlines()
            quote = random.choice(lines).strip()
            length = max(len(line) for line in quote.split("\\n"))
            formatted_quote = quote.replace("\\n", "\n")

            print("\n" + "="*length)
            print(formatted_quote)
            print("="*length + "\n")
    except FileNotFoundError:
        print("\nNo quotes file found. Exiting.\n")