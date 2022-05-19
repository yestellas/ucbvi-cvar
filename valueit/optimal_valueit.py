import math
import numpy as np

def optimal_value_iteration(mdp):
    n_states, n_actions = mdp.n_states, mdp.n_actions
    H = mdp.h
    pi_star = np.zeros([H, n_states, n_actions])
    v_star = np.zeros([H + 1, n_states])
    q_star = np.zeros([H+1, n_states, n_actions])
    next_q = np.zeros([n_states, n_actions])

    for i in range(H)[::-1]:
        for st in range(n_states):
            for ac in range(n_actions):
                q_sa = mdp.r[st, ac] + np.tensordot(mdp.p, v_star[i + 1], axes=([2], [0]))[st, ac]
                next_q[st, ac] = q_sa
        for st in range(n_states):
            best_ac, best_q = None, -math.inf
            for ac in range(n_actions):
                q_sa = next_q[st, ac]
                if best_q < q_sa:
                    best_q = q_sa
                    best_ac = ac
            pi_star[i, st, best_ac] = 1
            v_star[i, st] = best_q
        q_star[i] = next_q
    return v_star, q_star, pi_star
