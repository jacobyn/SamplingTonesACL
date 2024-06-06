# This implementation originates from
# @misc{grave2018unsupervised,
#       title={Unsupervised Alignment of Embeddings with Wasserstein Procrustes}, 
#       author={Edouard Grave and Armand Joulin and Quentin Berthet},
#       year={2018},
#       eprint={1805.11222},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }

# The GitHub repository is no longer findable, but implementations are reported to have been found in this archive:
# https://github.com/facebookresearch/MUSE?tab=readme-ov-file

import numpy as np
import ot
from sklearn.manifold import MDS


def objective(X, Y, R, o_reg=0.05, n=5000):
    Xn, Yn = X, Y
    C = -np.dot(np.dot(Xn, R), Yn.T)
    P = ot.sinkhorn(np.ones(len(X)), np.ones(len(Y)), C, o_reg, stopThr=1e-3)
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / (len(X) * len(Y))


def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


def align(X, Y, R, lr=10., bsz=200, nepoch=5, niter=1000,
        nmax=10000, reg=0.05, verbose=True, o_reg=0.05):
    bsz_X, bsz_Y = min(bsz, len(X)), min(bsz, len(Y))
    latest_loss = 0
    for epoch in range(1, nepoch + 1):
        for _it in range(1, niter + 1):
            # sample mini-batch
            xt = X[np.random.permutation(len(X))[:bsz_X], :]
            yt = Y[np.random.permutation(len(Y))[:bsz_Y], :]
            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)
            P = ot.sinkhorn(np.ones(bsz_X), np.ones(bsz_Y), C, reg, stopThr=1e-3)
            # compute gradient
            G = - np.dot(xt.T, np.dot(P, yt))
            R -= lr / bsz * G
            # project on orthogonal matrices
            U, s, VT = np.linalg.svd(R)
            R = np.dot(U, VT)
        bsz_X = min(int(bsz * 2), len(X))
        bsz_Y = min(int(bsz * 2), len(Y))
        niter //= 2
        latest_loss = objective(X, Y, R, o_reg, n=nmax)
    C = -np.dot(np.dot(X, R), Y.T)
    P = ot.sinkhorn(np.ones(len(X)), np.ones(len(Y)), C, o_reg, stopThr=1e-3)
    return P, R, latest_loss

def procrustes(X_src, Y_tgt):
    U, s, V = np.linalg.svd(np.dot(Y_tgt.T, X_src))
    return np.dot(U, V)

def convex_init(X, Y, niter=100, reg=0.05, apply_sqrt=False):
    nX, dX = X.shape
    nY, dY = Y.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([nY, nX]) / float(nX * nY)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        q = ot.sinkhorn(np.ones(nY), np.ones(nX), G, reg, numIterMax=1000, stopThr=1e-4)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    return procrustes(np.dot(P, X), Y).T

def master_ot(human_ratings_full, gpt_ratings_full, REG=0.5):
    maxload = 200000
    x_src = MDS().fit_transform(human_ratings_full.T)
    x_tgt = MDS().fit_transform(gpt_ratings_full.T)
    #human_table_corr, GPT_table_corr

    R0 = convex_init(x_src, x_tgt, niter=500, apply_sqrt=True, reg=REG)
    P, Q, latest_loss = align(x_src, x_tgt, R0.copy(), bsz=10, nmax=x_src.shape[0], lr=10, reg=REG, o_reg=REG, nepoch=15)


    ot_human_MDS = x_src
    aligned_GPT_MDS = P @ x_tgt @ Q.T 

    overall_MDS = MDS().fit_transform(np.vstack([ot_human_MDS, aligned_GPT_MDS]))
    ot_human_MDS, aligned_GPT_MDS = overall_MDS[:len(overall_MDS) // 2], overall_MDS[len(overall_MDS) // 2:]
    return ot_human_MDS, aligned_GPT_MDS, latest_loss