import numpy as np
import sys

def ccc(x,y, ignore=-5.0):
    """
        y_true: shape of (N, )
        y_pred: shape of (N, )
        """

    #y = y.reshape(-1)
    #x = x.reshape(-1)
    #index = y != ignore
    #y = y[index]
    #x = x[index]

    if len(y) <= 1:
        sys.exit()
        return 0.0
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)) +1e-8)
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/((x_s**2 + y_s**2 + (x_m - y_m)**2)+ 1e-8)
    return ccc




    # old code
    #batch_size = len(y_pred)
    #x_m = np.mean(y_pred)
    #y_m = np.mean(y_true)

    #x_std = np.std(y_true)
    #y_std = np.std(y_pred)

    #v_true = y_true - y_m
    #v_pred = y_pred - x_m

    #s_xy = np.sum(v_pred * v_true)

    #numerator = 2 * s_xy
    #denominator = x_std ** 2 + y_std ** 2 + (x_m - y_m) ** 2 + 1e-8

    #print(numerator)
    #print(denominator)

    #c = numerator / (denominator * batch_size)
    #print(c)
    #c = np.mean(c)
    #sys.exit()
    return c


def cccva(y_true, y_pred, ignore=-5.0):
    """
        y_true: shape of (N, 2)
        y_pred: shape of (N, 2)
        """
    v_pred = y_pred[:, 0]
    v_ture = y_true[:, 0]

    a_pred = y_pred[:,1]
    a_ture = y_true[:,1]

    ccc_v = ccc(v_ture, v_pred, ignore)
    ccc_a = ccc(a_ture, a_pred, ignore)

    return ccc_v,ccc_a,(ccc_v+ccc_a)/2


def ccc_numpy(y_true, y_pred):
    '''Reference numpy implementation of Lin's Concordance correlation coefficient'''

    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0, 1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2 + 1e-8)

    return ccc


class CCCMetric(object):
    def __init__(self,ignore_index = -5.0):
        self.y_pred = []
        self.y_true = []
        self.ignore = ignore_index

    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def clear(self):
        self.y_true = []
        self.y_pred = []

    def get(self):
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        print(y_true.shape)
        print(y_pred.shape)
        sys.exit()
        return ccc(y_true, y_pred,ignore=self.ignore)

if __name__ == '__main__':
    meric_ccc = CCCMetric()
    for i in range(10):
        logit = np.random.randn(16, 2)
        labels = np.random.randn(16, 2)
        meric_ccc.update(logit, logit)
        c = cccva(logit, logit)
        print(c)
        # print(ccc_numpy(logit, labels))
    print(meric_ccc.get())
