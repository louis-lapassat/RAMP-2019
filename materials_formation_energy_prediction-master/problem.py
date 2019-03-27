# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import KFold
import pymatgen as mg

from rampwf.score_types.base import BaseScoreType
from rampwf.utils.importing import import_file

problem_title = "Materials formation energy prediction"
target_column_name = "formation_energy_per_atom"
atom_groups = [
    ("Li", "O"),
    ("Cu", "Zn"),
    ("Cr", "Zn"),
    ("Co", "Zn"),
    ("Cu", "Ge"),
    ("Cu", "Si"),
    ("Al", "Cu"),
    ("Cu", "Mg"),
    ("Fe", "O"),
    ("Fe", "Zn"),
]
# this is the output size for making the setup
# work with the RAMP
# y_pred will have shape (n_examples, nb_atom_groups * 2)
# because we have one prediction per model (and we have nb_atom_groups models)
# we also attach to y_pred a mask of shape (n_examples, nb_atom_groups) which
# says for each example which regressor is responsible for predicting it
output_size = len(atom_groups) * 2
Predictions = rw.prediction_types.regression.make_regression(
    #this is also to make the setup work with the RAMP
    label_names=[None] * output_size
)


class Workflow(object):
    
    def __init__(self, atom_groups):
        self.atom_groups = atom_groups
        self.element_names = ["feature_extractor", "regressor"]

    def train_submission(self, module_path, X_df, y, train_is=None):
        fe_module = import_file(module_path, "feature_extractor")
        reg_module = import_file(module_path, "regressor")
        feature_extractor = fe_module.FeatureExtractor()
        #apply the feature extractor on all examples
        X = feature_extractor.transform(X_df)
        # the "groundtruth" y is normally 1D but we make it of shape
        #(n_examples, nb_atom_groups*2) filled with zeros except
        #in y[:, 0]. This is agian to make the setup work with the RAMP
        # (y and ypred must have the same shape)
        y = y[:, 0]

        # get compositions of each formula (example)
        compositions = X_df.formula.apply(mg.Composition).values
        compositions = [set(map(str, comp.keys())) for comp in compositions]
        #instantiate len(self.atom_groups) regressors
        regressors = [reg_module.Regressor() for _ in range(len(self.atom_groups))]
        #fit regressors
        for atoms, reg in zip(self.atom_groups, regressors):
            atoms = set(atoms)
            # test data is where all atoms of the group are contained
            # train data is everything else
            is_test = np.array(
                [atoms.issubset(comp) for comp in compositions]
            )
            is_train = ~is_test
            reg.fit(X[is_train], y[is_train])
        return feature_extractor, regressors

    def test_submission(self, trained_model, X_df):
        feature_extractor, regressors = trained_model
        #apply the feature extractor on all examples
        X = feature_extractor.transform(X_df)
        #get compositions of each formula (example)
        compositions = X_df.formula.apply(mg.Composition).values
        compositions = [set(map(str, comp.keys())) for comp in compositions]
        #get the predictions
        # preds is of shape (n_examples, len(self.atom_groups))
        #so each example has as much predictions as the number of regressors
        # (which is the number of atom groups)
        preds = np.array([regressor.predict(X).flatten() for regressor in regressors]).T
        #the mask will tell us for each example which regressor we should use
        #th mask is used in MAE to get the scores
        mask = np.zeros_like(preds)
        for i, (atoms, reg) in enumerate(zip(self.atom_groups, regressors)):
            atoms = set(atoms)
            is_test = np.array(
                [atoms.issubset(comp) for comp in compositions]
            )
            mask[is_test, i] = 1
        # the final prediction contain the actual predictions of the regressors
        # plus the mask
        preds = np.concatenate((preds, mask), axis=1)
        return preds


class MAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, ids=None, name="mae", precision=2):
        # ids are integers specifying which atom_groups we
        # evaluate
        if ids is None:
            ids = list(range(len(atom_groups)))
        self.ids = ids
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_true = y_true[:, 0]
        # get preds and mask
        y_pred_regs = y_pred[:, 0:len(atom_groups)]
        mask = y_pred[:, len(atom_groups):]
        scores = []
        for id_ in self.ids:
            #get preds and mask for each atom group we want
            #to evaluate on
            y_pred = y_pred_regs[:, id_]
            mask_ = mask[:, id_].astype(bool)
            score = np.mean(np.abs(y_true[mask_] - y_pred[mask_]))
            scores.append(score)
        return np.mean(scores)


workflow = Workflow(atom_groups)

score_types = []
score_types.append(MAE(name="mae"))
for i, atoms in enumerate(atom_groups):
    name = "mae_" + ",".join(atoms)
    score_types.append(MAE([i], name=name))


def get_cv(X, y):
    # no CV is done in this setup.
    # it is normal to see Nans in train scores
    #when doing ramp_test_submission
    indices = np.arange(len(X))
    return [[[], indices]]


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, "data", f_name))
    # y_array is padded with zeros to become of shape
    # (n_examples, output_size) to match y_pred
    # see above for explanation
    y = np.zeros((len(data), output_size))
    y[:, 0] = data[target_column_name].values
    X = data.drop([target_column_name], axis=1)
    return X, y


def get_train_data(path="."):
    f_name = "train.csv"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.csv"
    return _read_data(path, f_name)
