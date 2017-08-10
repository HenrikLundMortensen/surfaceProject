import numpy as np

def randomgrid(n):

    no = 3*n-9
    nag = n**2 - 3*n + 2
    nh = 7

    a = np.array(nh*[0] + no* [2] + nag *[1])
    np.random.shuffle(a)
    b = a.reshape(n,n)
    return b




