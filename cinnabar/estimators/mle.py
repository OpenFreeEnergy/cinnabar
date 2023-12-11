import numpy as np
from typing import Hashable, Optional

from . import BaseEstimator
from .. import FEMap, Measurement, ReferenceState

_REF = ReferenceState()


def _abs_label(mes: Measurement) -> Optional[Hashable]:
    # if measurement is absolute, return the other label
    # otherwise return None
    if mes.labelA == _REF:
        return mes.labelB
    elif mes.labelB == _REF:
        return mes.labelA
    else:
        return None


class MLEEstimator(BaseEstimator):
    def __init__(self):
        pass

    def estimate(self, prior: FEMap) -> FEMap:
        """Perform MLE estimate

        Parameters
        ----------
        prior : FEMap
          the existing FEMap.  Any computational results will be used

        Returns
        -------
        mle : FEMap
          a new FEMap containing the new edges
        """
        # 1) identify computational measurements to work on
        # this (potentially) includes both relative and absolute
        # TODO: Should also filter out non-absolute zero absolute values
        #       i.e. all absolute values need to be against the same reference
        #       point
        # TODO: Need to check assumption that these measurements form a weakly
        #       connected graph
        #       not clear if splitting into multiple connected graphs is the job
        #       of this estimator, or if splitting into connected graphs is a
        #       method for the FEMap object
        measurements = [
            m for m in prior
            if m.computational
        ]

        # 2) assign indices to the labels of these measurements
        # contains mapping of ligand names to integers
        # these integers will be their rank in the matrices
        label2id = dict()
        for m in measurements:
            for label in (m.labelA, m.labelB):
                if isinstance(label, ReferenceState):
                    continue
                if label in label2id:
                    continue
                label2id[label] = len(label2id)

        # consistently use whatever units the first measurement is in
        # output is also in this unit
        u = measurements[0].DG.u

        # 3) form edge matrices
        N = len(label2id)
        f_ij = np.zeros((N, N))
        df_ij = np.zeros((N, N))

        for m in measurements:
            if (abs_label := abs_label(m)) is not None:
                # masquerade absolute values as self-edges
                i = j = label2id[abs_label]
            else:
                i = label2id[m.labelA]
                j = label2id[m.labelB]

            if f_ij[i, j]:
                raise ValueError("Currently can't handle multiple values for "
                                 "a single edge")

            DG = m.DG.to(u).m
            dDG = m.uncertainty.to(u).m

            # these are DG, so anti-symmetrised
            # set this way round to absolute self-edge is positive
            # i.e. when i==j: f[i, i] = + m.DG.m
            f_ij[j, i] = - DG
            f_ij[i, j] = DG

            # these are uncertainties, so symmetrised
            df_ij[i, j] = dDG
            df_ij[j, i] = dDG

        # precompute df_ij ^ -2
        # some values are zero, so silence the warnings then zero these
        with np.errstate(divide='ignore'):
            df_ij2 = df_ij ** -2
        df_ij2[np.isinf(df_ij2)] = 0.0

        # 4) form F matrix (Fisher information matrix)
        # Fij :=
        # + theta_i^{-2} + \sum_{k!=i}{theta_{ik}^{-2}}  if i == j
        # - theta_{ij}^{-2}                              if i != j
        # i != j case:
        F_matrix = - df_ij2
        # i == j case:
        F_matrix[np.diag_indices_from(F_matrix)] = df_ij2.sum(axis=0)

        # 5) form Z matrix
        # z_i = theta_i^{-2} x_i + \sum_{j != i}{theta_{ij}^{-2} x_{ij}}
        # can use df_ij2 which is 0 for non-contributing entities
        z = (f_ij * df_ij2).sum(axis=0)

        # Compute MLE estimate (Eq 2)
        Finv = np.linalg.pinv(F_matrix)
        f_i = np.matmul(Finv, z)
        df_i = Finv.diagonal() ** 0.5

        # create reverse lookup to convert matrix indices back to ligand names
        # this list works as we know our indices are 0..n sequentially
        id2label = sorted(label2id.keys(), key=lambda x: label2id[x])

        # put the computed values into a new map and return this
        fem = FEMap()
        # a custom reference point that the MLE values are against
        g = ReferenceState(label='MLE')
        for name, MLE_f, MLE_df in zip(id2label, f_i, df_i):
            fem.add_measurement(
                Measurement(labelA=g,
                            labelB=name,
                            DF=MLE_f * u,
                            uncertainty=MLE_df * u,
                            computational=True,
                            source='MLE',
                )
            )

        # TODO: Add edge from true zero to MLE zero based on expt. values,
        #       assuming that the MLE values have same mean as expt. values

        return fem
