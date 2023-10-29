import numpy as np


def boxfilter(imSrc, r):
    hei, wid = imSrc.shape
    imDst = np.zeros_like(imSrc, dtype=np.float32)

    # Cumulative sum over Y axis
    imCum = np.cumsum(imSrc, axis=0)
    
    # Difference over Y axis
    imDst[0:r, :] = imCum[r:2*r, :]
    imDst[r+1:hei-r, :] = imCum[2*r+1:, :] - imCum[:hei-2*r-1, :]
    imDst[hei-r:, :] = np.tile(imCum[-1, :], (r, 1)) - imCum[hei-2*r-1:hei-r-1, :]
    
    # Cumulative sum over X axis
    imCum = np.cumsum(imDst, axis=1)
    
    # Difference over X axis
    imDst[:, 0:r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:wid-r] = imCum[:, 2*r+1:wid] - imCum[:, :wid-2*r-1]
    colonne_wid = imCum[:, -1]
    imDst[:, wid-r:wid] = np.column_stack([colonne_wid] * r) - imCum[:, wid-2*r-1:wid-r-1]
    
    return imDst

def composante_R(I):
    hei, wid, _ = I.shape
    N = np.ones((hei, wid))
    for i in range(hei):
        for j in range(wid):
            N[i,j] = I[i,j,0]
    return N

def composante_G(I):
    hei, wid, _ = I.shape
    N = np.ones((hei, wid))
    for i in range(hei):
        for j in range(wid):
            N[i,j] = I[i,j,1]
    return N

def composante_B(I):
    hei, wid, _ = I.shape
    N = np.ones((hei, wid))
    for i in range(hei):
        for j in range(wid):
            N[i,j] = I[i,j,2]
    return N

def is_null(A):
    hei, wid = A.shape
    for i in range(hei):
        for j in range(wid):
            if (A[i,j] == 0):
                return True
    return False

def guided_filter(I, p,r,eps):
    hei, wid, _ = I.shape
    N = boxfilter(np.ones((hei, wid)),r)
    print(N)
    print(is_null(N))
    small_value = 1e-10
    N = N + small_value 

    Ir = composante_R(I)
    Ig = composante_G(I)
    Ib = composante_B(I)

    mean_I_r = boxfilter(Ir, r) / N
    mean_I_g = boxfilter(Ig, r) / N
    mean_I_b = boxfilter(Ib, r) / N

    mean_p = boxfilter(p, r) / N

    mean_Ip_r = boxfilter(Ir*p, r) / N
    mean_Ip_g = boxfilter(Ig*p, r) / N
    mean_Ip_b = boxfilter(Ib*p, r) / N

    # covariance of (I, p) in each local patch.
    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p

    #variance of I in each local patch: the matrix Sigma in Eqn (14).
    # Note the variance in each local patch is a 3x3 symmetric matrix:
    #           rr, rg, rb
    #   Sigma = rg, gg, gb
    #           rb, gb, bb

    var_I_rr = boxfilter(Ir*Ir, r) / N - mean_I_r *  mean_I_r
    var_I_rg = boxfilter(Ir*Ig, r) / N - mean_I_r *  mean_I_g
    var_I_rb = boxfilter(Ir*Ib, r) / N - mean_I_r *  mean_I_b
    var_I_gg = boxfilter(Ig*Ig, r) / N - mean_I_g *  mean_I_g
    var_I_gb = boxfilter(Ig*Ib, r) / N - mean_I_g *  mean_I_b
    var_I_bb = boxfilter(Ib*Ib, r) / N - mean_I_b *  mean_I_b

    A = np.zeros((hei, wid, 3))
    for y in range(hei):
        for x in range(wid):
            Sigma = np.array([[var_I_rr[y, x], var_I_rg[y, x], var_I_rb[y, x]],
            [var_I_rg[y, x], var_I_gg[y, x], var_I_gb[y, x]],
            [var_I_rb[y, x], var_I_gb[y, x], var_I_bb[y, x]]])

            cov_Ip = [cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]]
            A[y, x,:] = np.dot(cov_Ip, np.linalg.inv(Sigma + eps * np.eye(3)))

    Ar = composante_R(A)
    Ag = composante_G(A)
    Ab = composante_B(A)
    b = mean_p - Ar * mean_I_r - Ag* mean_I_g - Ab * mean_I_b

    q = boxfilter(Ar, r)* Ir + boxfilter(Ag, r)* Ig+ boxfilter(Ab, r)* Ib+ boxfilter(b, r) / N

    return q
    

