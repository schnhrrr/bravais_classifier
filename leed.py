#%%

import numpy as np
import matplotlib.pyplot as plt

bravais_lattices = ['oblique', 'rect', 'centered_rect', 'square', 'hexagonal']#

def calc_a2_from_ratio(a1_len, ratio_min=0.1, ratio_max=3.0):
    r = np.random.uniform(ratio_min, ratio_max)   # r = a1/a2
    return a1_len / r

def sample_lattice_vectors(L=None, a_min=0.5e-10, a_max=5.0e-10):
    
    if L is None:
        L = np.random.choice(bravais_lattices)
    if L == 'oblique':
        a1_len = np.random.uniform(a_min, a_max)
        a2_len = calc_a2_from_ratio(a1_len)
        theta = np.random.uniform(30, 150) * np.pi / 180 # in radians

    if L == 'rect':
        a1_len = np.random.uniform(a_min, a_max)
        a2_len = calc_a2_from_ratio(a1_len)
        theta = np.pi / 2

    if L == 'square':
        a1_len = np.random.uniform(a_min, a_max)
        a2_len = a1_len
        theta = np.pi / 2
    
    if L == 'centered_rect':
        # calc rectangular lattice first
        A1_len = np.random.uniform(a_min, a_max)
        A2_len = calc_a2_from_ratio(A1_len)

        # convert to centered rectangular
        a1_len = A1_len
        a2_len = np.sqrt(A1_len**2 + A2_len**2)/2
        theta = np.arctan2(A2_len, A1_len)

    if L == 'hexagonal':
        a1_len = np.random.uniform(a_min, a_max)
        a2_len = a1_len
        theta = np.pi / 3

    a1_vec = np.array([a1_len, 0.0])
    a2_vec = a2_len * np.array([np.cos(theta), np.sin(theta)])
    return a1_vec, a2_vec, L

def reciprocal_lattice_vectors(a1, a2):
    R = np.array([[0,-1],[1,0]])
    normalization = np.dot(a1, R@a2)
    b_i = lambda a: 2*np.pi * R @ a / normalization
    return b_i(a1), b_i(a2)

def reciprocal_lattice_points(b1, b2, nmax=4, include_origin=False):
    h_vec = range(-nmax, nmax+1) 
    k_vec = range(-nmax, nmax+1)
    G_vec = []
    for h in h_vec:
        for k in k_vec:
            if not include_origin and h==0 and k==0:
                continue
            G_vec.append(h*b1 + k*b2)
    return np.array(G_vec)