import numpy as np

def GaussianSmoothingFilter(shape=(3,3),sigma=0.5):
    """
    Resultados comparados com funcao fspecial do matlab
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

#chamando funcao do gauss
GaussianSmoothingFilter((5,5),1)
#nesse caso estou criando um flitro gaussiano 5x5
