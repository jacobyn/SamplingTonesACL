# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This file involves implementations from
# @inproceedings{ruder2018,
#   author    = {Ruder, Sebastian  and  Cotterell, Ryan  and  Kementchedjhieva, Yova and S{\o}gaard, Anders},
#   title     = {A Discriminative Latent-Variable Model for Bilingual Lexicon Induction},
#   booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
#   year      = {2018}
# }
# @inproceedings{artetxe2018acl,
#   author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
#   title     = {A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings},
#   booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
#   year      = {2018},
#   pages     = {789--798}
# }

# @inproceedings{artetxe2018aaai,
#   author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
#   title     = {Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations},
#   booktitle = {Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence},
#   year      = {2018},
#   pages     = {5012--5019}
# }

# @inproceedings{artetxe2017acl,
#   author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
#   title     = {Learning bilingual word embeddings with (almost) no bilingual data},
#   booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
#   year      = {2017},
#   pages     = {451--462}
# }

# @inproceedings{artetxe2016emnlp,
#   author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
#   title     = {Learning principled bilingual mappings of word embeddings while preserving monolingual invariance},
#   booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
#   year      = {2016},
#   pages     = {2289--2294}
# }


import numpy as np
import pandas as pd
import sys
import time
import collections
import embedding_sebas as embeddings
from lap import lapmod

# TODO: Hyperparametrize everything here later, including plt.show() or not
METRIC = "cosine"

def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]


def mean_center(matrix):
    avg = np.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    avg = np.mean(matrix, axis=1)
    matrix -= avg[:, np.newaxis]


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)

def get_resorted_table(tbl, ind):
    new_indices = pd.concat(
        [
            ind,
            pd.Series(range(40))[~pd.Series(range(40)).isin(ind)]
        ]
    ).reset_index()[0]
    return tbl.iloc[:, new_indices]

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        mask = np.random.rand(*m.shape) >= p
        return m*mask

def lat_var(np, sims, n_similar, n_repeats, batch_size, asym):
    """
    Run the matching in the E-step of the latent-variable model.
    :param np: numpy or cupy, depending whether we run on CPU or GPU.
    :param xw: the xw matrix
    :param zw: the zw matrix
    :param sims: an matrix of shape (src_size, trg_size) where the similarity values
                 between each source word and target words are stored
    :param best_sim_forward: an array of shape (src_size), which stores the best similarity
                             scores for each
    :param n_similar:
    :param n_repeats:
    :param batch_size:
    :param asym:
    :return:
    """
    src_size = sims.shape[0]
    cc = np.empty(src_size * n_similar)  # 1D array of all finite elements of the assignement cost matrix
    kk = np.empty(src_size * n_similar)  # 1D array of the column indices. Must be sorted within one row.
    ii = np.empty((src_size * n_repeats + 1,), dtype=int)   # 1D array of indices of the row starts in cc.
    ii[0] = 0
    # if each src id should be matched to trg id, then we need to double the source indices
    for i in range(1, src_size * n_repeats + 1):
        ii[i] = ii[i - 1] + n_similar
    for i in range(0, src_size, batch_size):
        # j = min(x.shape[0], i + batch_size)
        j = min(i + batch_size, src_size)
        sim = sims[i:j]

        trg_indices = np.argpartition(sim, -n_similar)[:, -n_similar:]  # get indices of n largest elements
        if np != np:
            trg_indices = np.asnumpy(trg_indices)
        trg_indices.sort()  # sort the target indices

        trg_indices = trg_indices.flatten()
        row_indices = np.array([[i] * n_similar for i in range(j-i)]).flatten()
        sim_scores = sim[row_indices, trg_indices]
        costs = 1 - sim_scores
        if np != np:
            costs = np.asnumpy(costs)
        cc[i * n_similar:j * n_similar] = costs
        kk[i * n_similar:j * n_similar] = trg_indices
    if n_repeats > 1:
        # duplicate costs and target indices
        new_cc = cc
        new_kk = kk
        for i in range(1, n_repeats):
            new_cc = np.concatenate([new_cc, cc], axis=0)
            if asym == '1:2':
                # for 1:2, we don't duplicate the target indices
                new_kk = np.concatenate([new_kk, kk], axis=0)
            else:
                # update target indices so that they refer to new columns
                new_kk = np.concatenate([new_kk, kk + src_size * i], axis=0)
        cc = new_cc
        kk = new_kk
    # trg indices are targets assigned to each row id from 0-(n_rows-1)
    cost, trg_indices, _ = lapmod(src_size * n_repeats, cc, ii, kk)
    src_indices = np.concatenate([np.arange(src_size)] * n_repeats, 0)
    src_indices, trg_indices = np.asarray(src_indices), np.asarray(trg_indices)

    # remove the pairs in which a source word was connected to a target
    # which was not one of its k most similar words
    wrong_inds = []
    for i, trgind in enumerate(trg_indices):
        krow = ii[i]
        candidates = kk[krow:krow + n_similar]
        if trgind not in candidates:
            wrong_inds.append(i)
    trg_indices = np.delete(trg_indices, wrong_inds)
    src_indices = np.delete(src_indices, wrong_inds)

    for i in range(len(src_indices)):
        src_idx, trg_idx = src_indices[i], trg_indices[i]
        # we do this if args.n_repeats > 0 to assign the target
        # indices in the cost matrix to the correct idx
        while trg_idx >= src_size:
            # if we repeat, we have indices that are > n_rows
            trg_idx -= src_size
            trg_indices[i] = trg_idx
    return src_indices, trg_indices

def induce_one_side(
        source_ratings,
        target_ratings,
        direction,
        csls_neighborhood,
        translation_csls_neighborhood,
        n_induced_entries=1,
        show_plots=False,
        get_csls=False,
        return_embeddings=False
    ):
    src_words, x = list(source_ratings.columns), source_ratings.values.T.copy()
    trg_words, z = list(target_ratings.columns), target_ratings.values.T.copy()
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    normalize_pattern = ['unit', 'center', 'unit']
    embeddings.normalize(x, normalize_pattern)
    embeddings.normalize(z, normalize_pattern)
    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if show_plots:
        print('Using unsupervised initialization...')
    sim_size = min(x.shape[0], z.shape[0]) #if args.unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], args.unsupervised_vocab)
    u, s, vt = np.linalg.svd(x[:sim_size], full_matrices=False)
    xsim = (u*s).dot(u.T)
    u, s, vt = np.linalg.svd(z[:sim_size], full_matrices=False)
    zsim = (u*s).dot(u.T)
    del u, s, vt
    xsim.sort(axis=1)
    zsim.sort(axis=1)
    embeddings.normalize(xsim, normalize_pattern)
    embeddings.normalize(zsim, normalize_pattern)
    sim = xsim.dot(zsim.T)
    knn_sim_fwd = topk_mean(sim, k=csls_neighborhood)
    knn_sim_bwd = topk_mean(sim.T, k=csls_neighborhood)
    sim -= knn_sim_fwd[:, np.newaxis]/2 + knn_sim_bwd/2
    if [direction] == 'forward':
        src_indices = np.arange(sim_size)
        trg_indices = sim.argmax(axis=1)
    elif direction == 'backward':
        src_indices = sim.argmax(axis=0)
        trg_indices = np.arange(sim_size)
    elif direction == 'union':
        src_indices = np.concatenate((np.arange(sim_size), sim.argmax(axis=0)))
        trg_indices = np.concatenate((sim.argmax(axis=1), np.arange(sim_size)))
    del xsim, zsim, sim
    xw = np.empty_like(x)
    zw = np.empty_like(z)
    dtype = "float64"
    batch_size = 5
    src_size = x.shape[0]
    trg_size = z.shape[0]
    simfwd = np.empty((batch_size, trg_size), dtype=dtype)
    simbwd = np.empty((batch_size, src_size), dtype=dtype)

    best_sim_forward = np.full(src_size, -100, dtype=dtype)
    src_indices_forward = np.arange(src_size)
    trg_indices_forward = np.zeros(src_size, dtype=int)
    best_sim_backward = np.full(trg_size, -100, dtype=dtype)
    src_indices_backward = np.zeros(trg_size, dtype=int)
    trg_indices_backward = np.arange(trg_size)
    knn_sim_fwd = np.zeros(src_size, dtype=dtype)
    knn_sim_bwd = np.zeros(trg_size, dtype=dtype)

    self_learning = True
    end = not self_learning
    stochastic_interval=3
    stochastic_multiplier=2
    src_reweight, trg_reweight = 0.5, 0.5
    dim_reduction=0
    n_similar, n_repeats = 3, 1
    asym = '1:1'
    threshold = 1e-6
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = stochastic_initial = 0.1
    t = time.time()
    while True:
        # Increase the keep probability if we have not improve in args.stochastic_interval iterations
        if it - last_improvement > stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, stochastic_multiplier*keep_prob)
            last_improvement = it

        # Update the embedding mapping
        if not end:  # orthogonal mapping
            u, s, vt = np.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            w = vt.T.dot(u.T)
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping

            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z

            # STEP 1: Whitening
            def whitening_transformation(m):
                u, s, vt = np.linalg.svd(m, full_matrices=False)
                return vt.T.dot(np.diag(1/s)).dot(vt)
            wx1 = whitening_transformation(xw[src_indices])
            wz1 = whitening_transformation(zw[trg_indices])
            xw = xw.dot(wx1)
            zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            wx2, s, wz2_t = np.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
            wz2 = wz2_t.T
            xw = xw.dot(wx2)
            zw = zw.dot(wz2)

            # STEP 3: Re-weighting
            xw *= s**src_reweight
            zw *= s**trg_reweight

            # STEP 4: De-whitening
            xw = xw.dot(wx2.T.dot(np.linalg.inv(wx1)).dot(wx2))
            zw = zw.dot(wz2.T.dot(np.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction
            if dim_reduction > 0:
                xw = xw[:, :dim_reduction]
                zw = zw[:, :dim_reduction]

        # Self-learning
        if end:
            break
        else:
            # Update the training dictionary
            sims = np.zeros((src_size, trg_size), dtype=dtype)
            if direction in ('forward', 'union'):
                if csls_neighborhood > 0:
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=csls_neighborhood, inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                    simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
                    simfwd[:j-i] = dropout(simfwd[:j-i], 1 - keep_prob)
                    #if not args.lat_var:
                        # we get a dimension mismatch here as lat_var may produce fewer seeds
                        #simfwd[:j-i].argmax(axis=1, out=trg_indices_forward[i:j])
                    sims[i:j] = simfwd
                #if args.lat_var:
                    # TODO check if we can save memory by not storing a large sims matrix
                src_indices_forward, trg_indices_forward = lat_var(np, sims, n_similar, n_repeats, batch_size, asym)
            if direction in ('backward', 'union'):
                if csls_neighborhood > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                    simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                    simbwd[:j-i] = dropout(simbwd[:j-i], 1 - keep_prob)
                    #if not args.lat_var:
                    #    simbwd[:j-i].argmax(axis=1,out=src_indices_backward[i:j])
                    sims[i:j] = simbwd
                #if args.lat_var:
                    # swap the order of the indices
                trg_indices_backward, src_indices_backward = lat_var(np, sims, n_similar, n_repeats, batch_size, asym)
            if direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif direction == 'union':
                src_indices = np.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = np.concatenate((trg_indices_forward, trg_indices_backward))
            # elif args.direction == 'intersection':
            #     fwd_pairs = zip(src_indices_forward, trg_indices_forward)
            #     bwd_pairs = zip(src_indices_backward, trg_indices_backward)
            #     src_indices, trg_indices = zip(*set(fwd_pairs).intersection(bwd_pairs))
            #     src_indices, trg_indices = np.array(src_indices), np.array(trg_indices)

            # Objective function evaluation
            if direction == 'forward':
                objective = np.mean(best_sim_forward).tolist()
            elif direction == 'backward':
                objective = np.mean(best_sim_backward).tolist()
            elif direction == 'union':
                objective = (np.mean(best_sim_forward) + np.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= threshold:
                last_improvement = it
                best_objective = objective

            # Logging
            if show_plots:
                duration = time.time() - t
                print(file=sys.stderr)
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                print('\t- Drop probability: {0:9.4f}%'.format(100 - 100*keep_prob), file=sys.stderr)

        t = time.time()
        it += 1
    knn_sim_translation = np.zeros(z.shape[0])
    translation = collections.defaultdict(int)
    
    if get_csls:
        def get_csls_in_domain(mat, word_arr, ind_arr):
            overall_csls = {}
            for i in range(0, mat.shape[0], batch_size):
                j = min(i + batch_size, mat.shape[0])
                knn_sim_translation[i:j] = topk_mean(mat[i:j].dot(mat.T), k=translation_csls_neighborhood, inplace=True)
            for i in range(0, len(ind_arr), batch_size):
                j = min(i + batch_size, len(ind_arr))
                similarities = 2*mat[ind_arr[i:j]].dot(mat.T) - knn_sim_translation
                for k in range(similarities.shape[0]):
                    overall_csls[word_arr[ind_arr[i+k]]] = similarities[k]
            return overall_csls
        
        return get_csls_in_domain(x, src_words, src_indices),\
            get_csls_in_domain(z, trg_words, trg_indices),\
            src_words, trg_words
            
    
    for i in range(0, z.shape[0], batch_size):
        j = min(i + batch_size, z.shape[0])
        knn_sim_translation[i:j] = topk_mean(z[i:j].dot(x.T), k=translation_csls_neighborhood, inplace=True)
    for i in range(0, len(src_indices), batch_size):
        j = min(i + batch_size, len(src_indices))
        similarities = 2*x[src_indices[i:j]].dot(z.T) - knn_sim_translation  # Equivalent to the real CSLS scores for NN
        if n_induced_entries > 1:
            nn = np.argpartition(similarities, -n_induced_entries, axis=1)[:, -n_induced_entries:]
            for k in range(j-i):
                # translation[src_indices[i+k]] = nn[k]
                translation[src_indices[i+k]] = nn[k]
        else:
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src_indices[i+k]] = nn[k]
    # print(src_indices)
    # print(trg_indices)
    # print(src_words)
    # print(np.array(src_words)[src_indices])
    if return_embeddings:
        return {
            "src": {src_words[src_indices[i]]: x[src_indices[i]] for i in range(len(src_indices))},
            "tgt": {trg_words[trg_indices[i]]: z[trg_indices[i]] for i in range(len(trg_indices))}
        }
    
    if n_induced_entries > 1:
        obtained_data = {
            src_words[froms]: [trg_words[tos_ind] for tos_ind in tos]
            for froms, tos in translation.items()
        }
        return obtained_data
    else:
        obtained_data = np.unique(np.array([[src_words[a], trg_words[b]] for a, b in translation.items()]), axis=0)
        return {
            "from": obtained_data.T[0],
            "to": obtained_data.T[1],
        }