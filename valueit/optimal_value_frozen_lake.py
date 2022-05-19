import math
import numpy as np
### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
import sys

np.set_printoptions(threshold=sys.maxsize)

np.set_printoptions(precision=3)

def optimal_value_iteration(mdp):

    I = 30
    K = 140
    T = 10
    Rmax = 2

    n_states, n_actions = mdp.n_states, mdp.n_actions
    H = mdp.h
    pi_star = np.zeros((H, n_states))
    v_star = np.zeros((n_states))
    q_star = np.zeros((n_states, n_actions))


    print("\n" + "-" * 25 + "\nBeginning Computing for nP\n" + "-" * 25)
    nP = mdp.p


    for s in range (n_states):
        i = s%(2*I+1) - I
        bi = T * Rmax / K * i
        v_star[s] = max(bi, 0)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    for t in range (H - 1, -1, -1):
        new_value_function = np.copy(v_star)
        q_star = np.sum(nP * v_star, axis = 2)
        new_value_function = np.amax(q_star, axis = 1)
        v_star = new_value_function
        pi_star[t] = np.argmax(q_star, axis = 1)
    print(pi_star.shape)
        
    # Get best policy
    '''
    q_value_function = np.sum(nP * value_function, axis = (3,4))
    print(q_value_function[q])
    policy = np.argmax(q_value_function, axis = 2)
    print(policy[q])
    '''

    return v_star, q_star, pi_star


def optimal_value_iteration2(mdp):
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
