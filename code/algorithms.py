# define nmf algorithms
# different kinds of NMF models
import numpy as np


Iter_step = 500
skip_step = 50
error = 1e-4
rng = np.random.RandomState(13)

class Base():

    def __init__(
            self,
            n_components,
            iter_step):
        self.n_components = n_components
        self.iter_step = iter_step

    def _loss(self, X, D, R):
        return None

    def _update(self, X, D, R):
        return None

    def _init(self, X):

        n_features, n_samples = X.shape
        avg = np.sqrt(X.mean() / self.n_components)
        D = avg * rng.randn(n_features, self.n_components)
        R = avg * rng.randn(self.n_components, n_samples)
        np.abs(D, out=D)
        np.abs(R, out=R)

        return D, R


    def fit(self, X):

        D, R = self._init(X)
        losses = [self._loss(X, D, R)]
        for i in range(self.iter_step):
            D, R = self._update(X, D, R)
            losses.append(self._loss(X, D, R))
            if i % skip_step == 0:
              print('step:{:>4},  {:>10}'.format(i, losses[-1]))
            crit = abs(losses[-1] - losses[-2]) / losses[-2]
            if crit < error:
              break

        return D, R


class NMF_withweight(Base):
    def __init__(
            self,
            n_components,
            iter_step=500):
        super().__init__(n_components, iter_step)

    def _loss(self, X, D, R):
        return None

    def _update_weightmatrix(self, X, D, R):
        return None

    def _update(self, X, D, R):
        # update w
        W = self._update_weightmatrix(X, D, R)
        # update D
        denominator_D = (W * D.dot(R)).dot(R.T)
        denominator_D[denominator_D == 0] = np.finfo(np.float32).eps
        D = D * ((W * X).dot(R.T)) / denominator_D
        # update R
        denominator_R = D.T.dot(W * D.dot(R))
        denominator_R[denominator_R == 0] = np.finfo(np.float32).eps
        R = R * (D.T.dot(W * X)) / denominator_R
        return D, R


class NMF_standard(NMF_withweight):
    # Standard NMF
    def _loss(self, X, D, R):
        return np.linalg.norm(X - D.dot(R)) ** 2

    def _update_weightmatrix(self, X, D, R):
        return 1


class NMF_L1norm(NMF_withweight):
    # L1 norm based NMF
    def _loss(self, X, D, R):
        return np.sum(np.abs(X - D.dot(R)))

    def _update_weightmatrix(self, X, D, R):

        eps = X.var() / D.shape[1]
        return 1 / (np.sqrt(np.square(X - D.dot(R))) + eps ** 2)

class NMF_L21norm(NMF_withweight):
    # L1,2 norm NMF
    def _loss(self, X, D, R):
        return np.sum(np.sqrt(np.sum(np.square(X - D.dot(R)), axis=0)))

    def _update_weightmatrix(self, X, D, R):
        return 1 / np.sqrt(np.sum(np.square(X - D.dot(R)), axis=0))


class NMF_L1NormR(Base):

    def __init__(
            self,
            n_components,
            iter_step=500,
            lambda_=128):
        self.lambda_=128
        super().__init__(n_components, iter_step)

    def _loss(self, X, D, R, E):
        return (np.linalg.norm(X - D.dot(R) - E) ** 2 + self.lambda_ * np.sum(np.abs(E)))

    def _update(self, X, D, R):
        # E
        E = X - D.dot(R)
        index_greater = E > self.lambda_ / 2
        E[index_greater] = E[index_greater] - self.lambda_ / 2
        index_less = E < -self.lambda_ / 2
        E[index_less] = E[index_less] + self.lambda_ / 2
        E[np.logical_not(np.logical_or(index_greater, index_less))] = 0
        # update D
        E_minus_X = E - X
        E_minus_X_dot_RT = (E_minus_X).dot(R.T)
        denominator_D = 2 * D.dot(R).dot(R.T)
        denominator_D[denominator_D == 0 ] = np.finfo(np.float32).eps
        D = (D * (np.abs(E_minus_X_dot_RT) - E_minus_X_dot_RT) /
             denominator_D)
        # update R
        DT_dot_E_minus_X = D.T.dot(E_minus_X)
        denominator_R = 2 * D.T.dot(D).dot(R)
        denominator_R[denominator_R == 0] = np.finfo(np.float32).eps
        R = (R * (np.abs(DT_dot_E_minus_X) - DT_dot_E_minus_X) /
             denominator_R)
        # normalize D and R
        # keepdims=True makes its shape to be [1, n_components]
        normalization = np.sqrt(np.sum(np.square(D), axis=0, keepdims=True))
        D = D / normalization
        R = R * normalization.T
        return D, R, E

    def fit(self, X):
        D, R = self._init(X)
        #print(D.shape, R.shape)
        losses = [self._loss(X, D, R, X - D.dot(R))]
        for i in range(self.iter_step):
            D, R, E = self._update(X, D, R)
            losses.append(self._loss(X, D, R, E))
            crit = abs(losses[-1] - losses[-2]) / losses[-2]
            if i % skip_step == 0:
              print('step:{:>4},  {:>10}'.format(i, losses[-1]))
            if crit < error:
              break
        return D, R

class HCNMF(Base):
# HCNMF
    def __init__(
            self,
            n_components,
            iter_step=500,
            alpha=0.001,
            beta=0.001):
        self.alpha = alpha
        self.beta = beta
        super().__init__(n_components, iter_step)

    def _loss(self, X, D, R):
        return np.sum(np.sqrt(1 + np.square(X - D.dot(R))) - 1)

    def _update(self, X, D, R):
        denominator = np.sqrt(1+np.linalg.norm(X - D.dot(R)))
        # update D by Armijo rule
        grad_D = (D.dot(R).dot(R.T) - X.dot(R.T)) / denominator
        D_updated = D - self.alpha * grad_D
        while self._loss(X, D_updated, R) > self._loss(X, D, R):
            self.alpha *= 0.5
            D_updated = D - self.alpha * grad_D
        D = D_updated
        # update D by Armijo rule
        grad_R = (D.T.dot(D).dot(R) - D.T.dot(X)) / denominator
        R_updated = R - self.beta * grad_R
        while self._loss(X, D, R_updated) > self._loss(X, D, R):
            self.beta *= 0.5
            R_updated = R - self.beta * grad_R
        R = R_updated
        return D, R
