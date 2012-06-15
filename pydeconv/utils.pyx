import numpy as np
cimport numpy as np
cimport cython


cdef extern from "math.h":
    double log(double)

# The global prior on gradient is a Lorentz distribution
# phi(x) = (a/pi)*(a^2 + x^2)^(-1)
#
cdef double pi_inv = 1.0/np.pi

cpdef double _log_phi(double x, double a):
    return log(a*pi_inv/(a*a + x*x))

cdef double _dlog_phi(double x, double a):
    return - 2.0 * x / (a*a + x*x)

cdef np.ndarray check_output(tuple shape, np.ndarray output, bint init=False):
    if output is None:
        output = np.zeros(shape,'float')
    elif shape[0] != output.shape[0] or shape[1] != output.shape[1]:
            raise ValueError("Shapes mismatch")
    elif init :
        output[:,:] = 0.0
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convolve2d(np.ndarray[np.float_t, ndim=2] input,
        np.ndarray[np.float_t, ndim=2] kernel,
        np.ndarray[np.float_t, ndim=2] output=None):
    """ Convolves input with kernel in spatial domain.
    """
    cdef:
        int nKk = kernel.shape[0]
        int nKl = kernel.shape[1]
        int nIi = input.shape[0]
        int nIj = input.shape[1]


        int i,j,k,l,ii,jj
        double oij
        int hKk = nKk//2  # used to center the kernel
        int hKl = nKk//2  # used to center the kernel

    output = check_output((<object>input).shape, output, False)
    
    for i in range(nIi):
        for j in range(nIj):
            oij = 0
            for k in range(nKk):
                ii = i - k + hKk
                if ii < 0:
                    ii = min(-ii, nIi)
                elif ii >= nIi:
                    ii = max(0, 2*nIi - 2 - ii)
                
                for l in range(nKl):
                    jj = j - l + hKl
                    if jj < 0:
                        jj = min(-jj, nIj)
                    elif jj >= nIj:
                        jj = max(0, 2*nIj - 2 - jj)                    
                    oij += kernel[k,l] * input[ii,jj]
            output[i,j] = oij
    return output
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grad_P(pshape , np.ndarray[np.float_t, ndim=2] L,
        np.ndarray[np.float_t, ndim=2] R,
        np.ndarray[np.float_t, ndim=2] output = None):
    """ computes the gradient of \| L * P - I \|_2^2
        with respect to P, '*' being the convlotion product.
        with R = L x P - I
    """
    cdef:
        int npk = pshape[0]
        int npl = pshape[1]

        int nli = L.shape[0]
        int nlj = L.shape[1]

        int i,j,k,l,ii,jj
        double pkl

        int hpk = npk//2
        int hpl = npl//2

    output = check_output(pshape, output)

    if R.shape[0] != nli or R.shape[1] != nlj:
        raise ValueError("Shapes mismatch")

    for k in range(npk):
        for l in range(npl):
            pkl = 0
            for i in range(nli):
                ii = i - k + hpk
                if ii < 0:
                    ii = min(-ii, nli)
                elif ii >= nli:
                    ii = max(0, 2*nli - 2 - ii)
                for j in range(nlj):
                    jj = j - l + hpl
                    if jj < 0:
                        jj = min(-jj, nlj)
                    elif jj >= nlj:
                        jj = max(0, 2*nlj - 2 - jj)
                    pkl += L[ii,jj]*R[i,j]
            output[k,l] = 2*pkl
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grad_L(np.ndarray[np.float_t, ndim=2] P,
        np.ndarray[np.float_t, ndim=2] R,
        np.ndarray[np.float_t, ndim=2] output = None):
    """ computes the gradient of \| L * P - I \|_2^2
        with respect to L, '*' being the convlotion product.
        with R = L * P - I
    """
    cdef:
        int npk = P.shape[0]
        int npl = P.shape[1]

        int nli = R.shape[0]
        int nlj = R.shape[1]

        np.ndarray[np.float_t, ndim=2] dL = np.zeros((<object>R).shape)

        int i,j,k,l,ii,jj
        double lij
        int hpk = npk//2
        int hpl = npl//2

    output = check_output((<object>R).shape, output)

    for i in range(nli):
        for j in range(nlj):
            lij = 0
            for k in range(npk):
                ii = i - k + hpk
                if ii < 0:
                    ii = min(-ii, nli)
                elif ii >= nli:
                    ii = max(0, 2*nli - 2 - ii)
                for l in range(npl):
                    jj = j - l + hpl
                    if jj < 0:
                        jj = min(-jj, nlj)
                    elif jj >= nlj:
                        jj = max(0, 2*nlj - 2 - jj)
                    lij += P[npk-k-1,npl-l-1]*R[ii,jj]
            output[i,j] = 2*lij
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def norm2(np.ndarray[np.float_t, ndim=2] mat, np.ndarray[np.uint8_t, ndim=2] mask=None):
    """ Computes || mat ||_2^2
    """
    cdef:
        int i,j
        int n = mat.shape[0]
        int m = mat.shape[1]
        double Iij, r = 0

    if mask is None:
        for i in range(n):
            for j in range(m):
                Iij = mat[i,j]
                r += Iij * Iij
    else:
        for i in range(n):
            for j in range(m):
                if mask[i,j] :
                    Iij = mat[i,j]
                    r += Iij * Iij
    return r
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dx(np.ndarray[np.float_t, ndim=2] I, np.ndarray[np.float_t, ndim=2] output=None):
    """ Computes the forward difference
        taking care of the borders
    """
    cdef:
        int i,j
        int n = I.shape[0]
        int m = I.shape[1]
        double diff, r = 0

    output = check_output((<object>I).shape, output)
    
    for i in range(n):
        for j in range(m-1):
            output[i,j] = I[i,j+1] - I[i,j]
        output[i,m-1] = - output[i, m-2]
    return output
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dx_b(np.ndarray[np.float_t, ndim=2] I, np.ndarray[np.float_t, ndim=2] output=None):
    """ Computes the backward difference
        taking care of the borders
    """
    cdef:
        int i,j
        int n = I.shape[0]
        int m = I.shape[1]
        double diff, r = 0

    output = check_output((<object>I).shape, output)

    for i in range(n):
        for j in range(1,m):
            output[i,j] = I[i,j] - I[i,j-1]
        output[i,0] = - output[i, 1]
    return output

@cython.boundscheck(False)
def dy(np.ndarray[np.float_t, ndim=2] I, np.ndarray[np.float_t, ndim=2] output=None):
    """ Computes the forward difference along y
        taking care of the borders
    """
    output = check_output((<object>I).shape, output)
    return dx(I.T, output.T).T

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dy_b(np.ndarray[np.float_t, ndim=2] I, np.ndarray[np.float_t, ndim=2] output=None):
    """ Computes the backward difference along y
        taking care of the borders
    """
    output = check_output((<object>I).shape, output)
    return dx_b(I.T, output.T).T

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def global_prior(np.ndarray[np.float_t, ndim=2] dL, double a):
    cdef:
        int i,j
        int n = dL.shape[0]
        int m = dL.shape[1]
        double r = 0

    for i in range(n):
        for j in range(m):
            r += _log_phi(dL[i,j], a)
    return r

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grad_global_prior_x(np.ndarray[np.float_t, ndim=2] dL, double a, np.ndarray[np.float_t, ndim=2] output=None):
    cdef:
        int i,j
        int n = dL.shape[0]
        int m = dL.shape[1]
        double r

    output = check_output((<object>dL).shape, output, True)

    for i in range(n):
        for j in range(m-1):
            r = _dlog_phi(dL[i,j], a)
            output[i,j] -= r
            output[i,j+1] += r
            
        r = _dlog_phi(dL[i, m-1], a)
        output[i,m-1] -= r
        output[i,m-2] -= r
    return output


def grad_global_prior_y(np.ndarray[np.float_t, ndim=2] dL, double a, np.ndarray[np.float_t, ndim=2] output=None):
    output = check_output((<object>dL).shape, output, True)
    return grad_global_prior_x(dL.T, a, output.T).T

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def local_prior(np.ndarray[np.float_t, ndim=2] I, np.ndarray[np.float_t, ndim=2] J, np.ndarray[np.uint8_t, ndim=2] M):
    """ computes the energy for the local prior
    """
    cdef:
        int i,j
        int n = I.shape[0]
        int m = I.shape[1]
        double diff, r = 0

    if n != J.shape[0] or m != J.shape[1]:
        raise ValueError("Shapes mismatch")

    for i in range(n):
        for j in range(m):
            if M[i,j]: 
                diff = I[i,j] - J[i,j]
                r += diff * diff
    return r
    
def grad_local_prior_x(np.ndarray[np.float_t, ndim=2] dxL,
                       np.ndarray[np.float_t, ndim=2] dxI,
                       np.ndarray[np.uint8_t, ndim=2] M,
                       np.ndarray[np.float_t, ndim=2] output=None):
    """ computes the energy for the local prior
    """
    output = check_output((<object>dxL).shape, output)
    dx_b(dxL - dxI, output)
    output *= -2.0
    return output * M

def grad_local_prior_y(np.ndarray[np.float_t, ndim=2] dyL,
                       np.ndarray[np.float_t, ndim=2] dyI,
                       np.ndarray[np.uint8_t, ndim=2] M,
                       np.ndarray[np.float_t, ndim=2] output=None):
    """ computes the energy for the local prior
    """
    output = check_output((<object>dyL).shape, output)
    return grad_local_prior_x(dyL.T, dyI.T, M.T, output.T).T
