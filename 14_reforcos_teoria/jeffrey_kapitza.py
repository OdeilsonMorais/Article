"""Coeficiente de segunda ordem (Jeffrey 1973) com resistência de interface.

k_eff/k_m = 1 + 3*beta*phi + K2*phi^2,  K2 = 3*beta^2 + 9*beta*I,
I = int_{rho_min}^inf fbar(rho) rho^2 drho,  rho = r/R,
fbar = média orientacional do excesso relativo do dipolo de uma esfera na
presença de outra (campo axial: peso 1/3; transversal: peso 2/3).

Duas esferas iguais, expansão em multipolos gêmeos. A resistência de Kapitza
entra na polarizabilidade de ordem n: kappa_n = kappa/(1 + n*alpha*kappa),
lambda_n = n(kappa_n - 1)/(n*kappa_n + n + 1), alpha = R_K k_m/R.
Os coeficientes de re-expansão são obtidos por projeção numérica (Gauss-Legendre).

A extensão com resistência de interface é de Chiew e Glandt, Chem. Eng. Sci. 42
(1987) 2677. O eta deles é k_p/(h R) = alpha*kappa. Validação (12/11/2026),
kappa = 1000, rho_min = 2: eta = 0, 50, 100, 500 -> K2 = 4.4795, 3.0472, 2.1716,
0.1963 (Tabela 1 deles: 4.47, 3.047, 2.172, 0.196).
"""
import numpy as np
from scipy.special import lpmv
from scipy.integrate import quad

def lam(n, kappa, alpha):
    if np.isinf(kappa):
        kn = np.inf if alpha == 0 else 1.0/(n*alpha)
    else:
        kn = kappa/(1.0 + n*alpha*kappa)
    if np.isinf(kn):
        return 1.0
    return n*(kn - 1.0)/(n*kn + n + 1.0)

def dipolo_relativo(rho, kappa, alpha, m, N):
    """A_1/A_1(isolada) para duas esferas com centros a distância rho (em R)."""
    nq = max(3*N, 120)
    x, w = np.polynomial.legendre.leggauss(nq)        # x = cos(theta1)
    sin1 = np.sqrt(1 - x*x)
    # ponto na superfície da esfera 1 (centro na origem), esfera 2 em z = rho
    z2 = x - rho; s2 = sin1
    r2 = np.sqrt(z2*z2 + s2*s2); c2 = z2/r2
    ns = np.arange(1, N+1)
    # c[s-1, n-1]: coef. de P_s^m(cos th1) na expansão de (1/r2)^{n+1} P_n^m(cos th2) em r1 = 1
    Ps = np.array([lpmv(m, s, x) for s in ns])                      # (N, nq)
    norm = np.array([2.0/(2*s+1)*np.prod(np.arange(s-m+1, s+m+1, dtype=float)) for s in ns])
    F = np.array([r2**(-(n+1))*lpmv(m, n, c2) for n in ns])         # (N, nq)
    C = (Ps*w) @ F.T / norm[:, None]                                 # (N, N)
    L = np.array([lam(n, kappa, alpha) for n in ns])
    # As duas esferas usam o mesmo eixo z. Pela simetria de espelho em
    # z = rho/2, os multipolos da esfera 2 são B_n = (-1)^(n+1) A_n, nos casos
    # axial e transversal.
    sig = (-1.0)**(ns + 1)
    # incidente em 1: a_s = amb_s + sum_n C[s,n] * sig_n * A_n ; A_s = -L_s a_s
    amb = np.zeros(N); amb[0] = 1.0
    M = np.eye(N) + L[:, None]*C*sig[None, :]
    A = np.linalg.solve(M, -L*amb)
    return A[0]/(-L[0])

def fbar(rho, kappa, alpha):
    gap = rho - 2.0
    N = int(min(NMAX, max(30, 10/np.sqrt(max(gap, 1e-5)))))
    fa = dipolo_relativo(rho, kappa, alpha, 0, N) - 1.0
    ft = dipolo_relativo(rho, kappa, alpha, 1, N) - 1.0
    return (fa + 2*ft)/3.0, fa, ft

NMAX = 250

def K2(kappa, alpha, rho_min=2.0):
    """Integra em painéis de Gauss-Legendre, espaçados em log na lacuna."""
    beta = lam(1, kappa, alpha)
    xg, wg = np.polynomial.legendre.leggauss(8)
    bordas = rho_min + np.concatenate(([0.0], np.logspace(-5, 1, 25)))   # até rho_min + 10
    I = 0.0
    for a, b in zip(bordas[:-1], bordas[1:]):
        r = 0.5*(b - a)*xg + 0.5*(b + a)
        I += 0.5*(b - a)*sum(w*fbar(ri, kappa, alpha)[0]*ri*ri for ri, w in zip(r, wg))
    I += 2*beta**2/(3*bordas[-1]**3)       # cauda: fbar ~ 2 beta^2/rho^6
    return 3*beta**2 + 9*beta*I, beta, 9*beta*I

if __name__ == "__main__":
    for rho in (6.0, 3.0):
        f, fa, ft = fbar(rho, 25.0, 0.0); b = lam(1, 25.0, 0.0)
        print(f"teste rho={rho}: fbar={f:.3e}  2beta^2/rho^6={2*b*b/rho**6:.3e}  fa={fa:.3e} (2b/rho^3={2*b/rho**3:.3e})  ft={ft:.3e}")
    print("Validação (Jeffrey 1973): kappa = inf -> 4.51 ; kappa = 0 -> 0.588")
    for kappa in (np.inf, 100.0, 10.0, 0.0):
        k2, beta, c = K2(kappa, 0.0)
        print(f"  kappa={kappa:>6}: beta={beta:+.4f}  K2={k2:.4f}  c=K2-3beta^2={c:+.4f}")
    print("\nNosso caso: k_p/k_m = 25, R = 50 nm")
    for rk in (1e-10, 1e-8):
        alpha = rk*0.2/50e-9
        for rmin in (2.0, 2.05):
            k2, beta, c = K2(25.0, alpha, rmin)
            print(f"  R_K={rk:g} alpha={alpha:g} rho_min={rmin}: beta={beta:.4f} K2={k2:.4f} c={c:.4f}")
