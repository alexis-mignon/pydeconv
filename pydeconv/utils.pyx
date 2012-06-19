import numpy as np
cimport numpy as np
cimport cython


cdef extern from "math.h":
    double log(double)
    double sin(double)
    double pow(double, double)
    double fabs(double)

# The global prior on gradient is a Lorentz distribution
# phi(x) = (a/pi)*(a^2 + x^2)^(-1)
#
cdef double _2pi_inv = 1.0/(2*np.pi)

cpdef double _log_phi(double x, double a, double b):
    return log(b*pow(a,b-1)*sin(np.pi/b)*_2pi_inv/(pow(a,b) + pow(fabs(x),b)))

cdef double _dlog_phi(double x, double a, double b):
    cdef:
        double apowb = pow(a,b)
        double absx = fabs(x)
        double den = (apowb + pow(absx,b))
        
    return - b*b*apowb/a*sin(np.pi/b)*_2pi_inv/(x*den*den)

cdef np.ndarray check_output(tuple shape, np.ndarray output, bint init=False):
    if output is None:
        output = np.zeros(shape,'float')
    elif len(shape) != output.ndim or \
         shape[0] != output.shape[0] or \
         shape[1] != output.shape[1] or \
         shape[2] != output.shape[2]:
            raise ValueError("Shapes mismatch")
    elif init :
        output[:,:] = 0.0
    return output

def check_intput_dim(np.ndarray input):
    if input.ndim == 2:
        return input[...,np.newaxis]
    else:
        return input

def check_output_dim(np.ndarray output):
    if output.ndim == 3 and output.shape[2] == 1:
        return output[...,0]
    else:
        return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def project_simplex(np.ndarray y, double norm=1.0):
    """ project a vector on the unit simplex
    """
    cdef :
        int i
        double ti = 0.0
        double cumyi = 0.0
        int n = y.size
        np.ndarray y_ = y.flatten()
        np.ndarray isort
        np.ndarray[np.float_t, ndim=1] ysort
        bint ok
        
    isort = y_.argsort()[::-1]
    ysort = y_[isort]
    ok = False
    
    for i in range(n-1):
        cumyi+= ysort[i]
        ti = (cumyi - norm)/(i+1)
        if ti >= ysort[i+1]:
            ok = True
            break
    
    if not ok:
        ti = (cumyi + ysort[n-1] - norm)/n

    return (y-ti).clip(0,np.inf)
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convolve2d(np.ndarray input,
        np.ndarray[np.float_t, ndim=2] kernel,
        np.ndarray output=None):
    """ Convolves input with kernel in spatial domain.
    """
    cdef:
        np.ndarray[np.float_t, ndim=3] intput_ = check_input_dim(output_dim)
        np.ndarray[np.float_t, ndim=3] output_

        int nKk = kernel.shape[0]
        int nKl = kernel.shape[1]
        int nIi = input_.shape[0]
        int nIj = input_.shape[1]
        int nc = input_.shape[2]

        int i,j,k,l,ii,jj,c
        double oij
        int hKk = nKk//2  # used to center the kernel
        int hKl = nKl//2  # used to center the kernel

    
    output = check_output((<object>input_).shape, output, False)
    output_ = check_input_dim(output)

    for c in range(nc):
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
                        oij += kernel[k,l] * input[ii,jj,c]
                output_[i,j,c] = oij
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grad_P(pshape , np.ndarray L,
        np.ndarray R,
        np.ndarray output = None):
    """ computes the gradient of \| L * P - I \|_2^2
        with respect to P, '*' being the convlotion product.
        with R = L x P - I
    """
    cdef:
        int npk = pshape[0]
        int npl = pshape[1]

        int nli = L.shape[0]
        int nlj = L.shape[1]
        int nc = L.shape[2]

        int i,j,k,l,ii,jj
        double pkl

        int hpk = npk//2
        int hpl = npl//2

        np.ndarray[np.float_t, ndim=3] input_ = check_intput_dim(intput)
        np.ndarray[np.float_t, ndim=3] output_
        

    output = check_output(pshape, output)
    output_ = check_intput_dim(output)

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
                    for c in range(nc):
                        pkl += L[ii,jj,c]*R[i,j,c]
            output[k,l,c] = 2*pkl
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grad_L(np.ndarray[np.float_t, ndim=2] P,
        np.ndarray[np.float_t, ndim=3] R,
        np.ndarray[np.float_t, ndim=3] output = None):
    """ computes the gradient of \| L * P - I \|_2^2
        with respect to L, '*' being the convlotion product.
        with R = L * P - I
    """
    cdef:
        int npk = P.shape[0]
        int npl = P.shape[1]

        int nli = R.shape[0]
        int nlj = R.shape[1]
        int nc = R.shape[2]

        np.ndarray[np.float_t, ndim=3] dL = np.zeros((<object>R).shape)

        int i,j,k,l,ii,jj,c
        double lij
        int hpk = npk//2
        int hpl = npl//2

    output = check_output((<object>R).shape, output)

    for c in range(nc):
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
                        lij += P[npk-k-1,npl-l-1]*R[ii,jj,c]
                output[i,j,c] = 2*lij
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def norm2(np.ndarray[np.float_t, ndim=3] mat, np.ndarray[np.uint8_t, ndim=2] mask=None):
    """ Computes || mat ||_2^2
    """
    cdef:
        int i,j,c
        int n = mat.shape[0]
        int m = mat.shape[1]
        int nc = mat.shape[2]
        double Iij, r = 0

    if mask is None:
        for i in range(n):
            for j in range(m):
                for c in range(nc):
                    Iij = mat[i,j,c]
                    r += Iij * Iij
    else:
        for i in range(n):
            for j in range(m):
                if mask[i,j] :
                    for c in range(nc):
                        Iij = mat[i,j,c]
                        r += Iij * Iij
    return r
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dx(np.ndarray[np.float_t, ndim=3] I, np.ndarray[np.float_t, ndim=3] output=None):
    """ Computes the forward difference
        taking care of the borders
    """
    cdef:
        int i,j,c
        int n = I.shape[0]
        int m = I.shape[1]
        int nc = I.shape[2]
        double diff, r = 0

    output = check_output((<object>I).shape, output)

    for c in range(nc):
        for i in range(n):
            for j in range(m-1):
                output[i,j,c] = I[i,j+1,c] - I[i,j,c]
            output[i,m-1, c] = - output[i, m-2, c]
    return output
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dx_b(np.ndarray[np.float_t, ndim=3] I, np.ndarray[np.float_t, ndim=3] output=None):
    """ Computes the backward difference
        taking care of the borders
    """
    cdef:
        int i,j,c
        int n = I.shape[0]
        int m = I.shape[1]
        int nc = I.shape[2]
        
        double diff, r = 0

    output = check_output((<object>I).shape, output)

    for c in range(nc):
        for i in range(n):
            for j in range(1,m):
                output[i,j] = I[i,j] - I[i,j-1]
            output[i,0] = - output[i, 1]
    return output

@cython.boundscheck(False)
def dy(np.ndarray[np.float_t, ndim=3] I, np.ndarray[np.float_t, ndim=3] output=None):
    """ Computes the forward difference along y
        taking care of the borders
    """
    output = check_output((<object>I).shape, output)
    return dx(I.swapaxes(0,1), output.swapaxes(0,1)).swapaxes(0,1)

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dy_b(np.ndarray[np.float_t, ndim=3] I, np.ndarray[np.float_t, ndim=3] output=None):
    """ Computes the backward difference along y
        taking care of the borders
    """
    output = check_output((<object>I).shape, output)
    return dx_b(I.swapaxes(0,1), output.swapaxes(0,1)).swapaxes(0,1)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def global_prior(np.ndarray[np.float_t, ndim=3] dL, double a, double b):
    cdef:
        int i,j,c
        int n = dL.shape[0]
        int m = dL.shape[1]
        int nc = dL.shape[2]
        double r = 0

    for c in range(nc):
        for i in range(n):
            for j in range(m):
                r += _log_phi(dL[i,j,c], a, b)
    return r
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grad_global_prior_x(np.ndarray[np.float_t, ndim=3] dL, double a, double b, np.ndarray[np.float_t, ndim=3] output=None):
    cdef:
        int i,j,c
        int n = dL.shape[0]
        int m = dL.shape[1]
        int nc = dL.shape[2]
        double r

    output = check_output((<object>dL).shape, output, True)

    for c in range(nc):
        for i in range(n):
            for j in range(m-1):
                r = _dlog_phi(dL[i,j,c], a, b)
                output[i,j,c] -= r
                output[i,j+1,c] += r
                
            r = _dlog_phi(dL[i, m-1, c], a, b)
            output[i,m-1,c] -= r
            output[i,m-2,c] -= r
    return output


def grad_global_prior_y(np.ndarray[np.float_t, ndim=3] dL, double a, double b, np.ndarray[np.float_t, ndim=3] output=None):
    output = check_output((<object>dL).shape, output, True)
    return grad_global_prior_x(dL.swapaxes(0,1), a,b, output.swapaxes(0,1)).swapaxes(0,1)

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def local_prior(np.ndarray[np.float_t, ndim=3] I, np.ndarray[np.float_t, ndim=3] J, np.ndarray[np.uint8_t, ndim=2] M):
    """ computes the energy for the local prior
    """
    cdef:
        int i,j,c
        int n = I.shape[0]
        int m = I.shape[1]
        int nc = I.shape[2]
        double diff, r = 0

    if n != J.shape[0] or m != J.shape[1] or nc != J.shape[2]:
        raise ValueError("Shapes mismatch")
   
    for i in range(n):
        for j in range(m):
            if M[i,j]:
                for c in range(nc):
                    diff = I[i,j,c] - J[i,j,c]
                    r += diff * diff
    return r
    
def grad_local_prior_x(np.ndarray[np.float_t, ndim=3] dxL,
                       np.ndarray[np.float_t, ndim=3] dxI,
                       np.ndarray[np.uint8_t, ndim=2] M,
                       np.ndarray[np.float_t, ndim=3] output=None):
    """ computes the energy for the local prior
    """
    output = check_output((<object>dxL).shape, output)
    dx_b(dxL - dxI, output)
    output *= -2.0
    return output * M

def grad_local_prior_y(np.ndarray[np.float_t, ndim=3] dyL,
                       np.ndarray[np.float_t, ndim=3] dyI,
                       np.ndarray[np.uint8_t, ndim=2] M,
                       np.ndarray[np.float_t, ndim=3] output=None):
    """ computes the energy for the local prior
    """
    output = check_output((<object>dyL).shape, output)
    return grad_local_prior_x(dyL.swapaxes(0,1), dyI.swapaxes(0,1), M.swapaxes(0,1), output.swapaxes(0,1)).swapaxes(0,1)
