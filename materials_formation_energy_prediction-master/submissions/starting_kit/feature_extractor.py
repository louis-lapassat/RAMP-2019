# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import defaultdict
import numpy as np
import re
import pymatgen as mg

elements = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
            'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

class FeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df):
        formulas = X_df.formula.values
        input = np.zeros(shape=(len(formulas), len(elements)), dtype=np.float32)
        for i, formula in enumerate(formulas):
            comp = mg.Composition(formula).as_dict()
            for k in comp.keys():
                input[i][elements.index(k)] = comp[k]
        return input
