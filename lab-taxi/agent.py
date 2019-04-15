import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, strategy="expected_sarsa", GLIE="iter_inverse", GLIE_param=None, alpha = 0.005, gamma = 1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - strategy: To be used to look for optimal policy. Can be:
            "sarsa", "sarsamax", "expected_sarsa"
        - GLIE: determines epsilon behavior per episode. Can be:
            "constant": epsilon has a constant value all the time, then GLIE_param
                        should contain the constant value of epsilon.
            "dec_bias": decrease linearly and stop at bias. GLIE_param
                        is then the arrival bias value of epsilon.
            "iter_inverse": epsilon  = 1 / iter_num. No need to pass GLIE_param
        - alpha: Learning rate
        - gamma: Discount factor

        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        # Configurtion dictionary
        self.conf = {'strategy' : strategy,
                     'GLIE' : GLIE,
                     'GLIE_param' : GLIE_param,
                     'alpha' : alpha,
                     'gamma' : gamma}

        # Function mappings into specific strategy methods
        self.comp = {'init_st' : '_init_st_' + self.conf['strategy'],
                     'GLIE_map' : '_GLIE_map_' + self.conf['GLIE'],
                     'step' : '_step_' + self.conf['strategy'],
                     'select_action' : '_select_action_' + self.conf['strategy']}

        # Memory: Saving agent's knowledge internally
        self.mem = {'st' : None,
                    'at' : None,
                    'rt_1' : None,
                    'st_1' : None,
                    'at_1' : None}

        # Variables used for epsilon control per episode
        self.num_episodes = None
        self.convergence_iters = None
        self.iter_num = 1
        self.epsilon = None

    def set_num_episodes(self, num_episodes):
        self.num_episodes = num_episodes
        self.convergence_iters = int(num_episodes * 7 / 8)

    def init_st(self, state):
        """ Given the initial state, initialize strategy. Particularly important for
        sarsa

        Params
        ======
        - state: the initial state of the environment
        """
        # Initialize epsilon
        GLIE_method = getattr(self, self.comp['GLIE_map'])
        self.epsilon = GLIE_method()

        init_method = getattr(self, self.comp['init_st'])
        init_method(state)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        select_action_method = getattr(self, self.comp['select_action'])
        return select_action_method(state)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.mem['st'] = state
        self.mem['at'] = action
        self.mem['rt_1'] = reward
        self.mem['st_1'] = next_state

        step_method = getattr(self, self.comp['step'])
        step_method()

        # Actualize number of iterations and the corresponding epsilon
        self.iter_num += 1
        GLIE_method = getattr(self, self.comp['GLIE_map'])
        self.epsilon = GLIE_method()


    # ------------------------------------------------------------------------
    # GENERAL INSTRUMENTAL FUNCTIONS----------------------------------------
    def _select_action_from_Q(self, state, Q, nA, epsilon):
        action = self._action_epsilon_greedy(Q[state], nA, epsilon)
        return action

    def _action_epsilon_greedy(self, Qs, nA, epsilon):
        if random.random() > epsilon: # select greedy action with probability epsilon
            return np.argmax(Qs)
        else:                     # otherwise, select an action randomly
            return random.choice(np.arange(nA))

    def _epsilon_greedy_state_probs(self, Qs, epsilon):
        policy_s = epsilon * np.ones(Qs.shape[0])/Qs.shape[0]
        max_index = np.argmax(Qs)
        policy_s[max_index] = 1 - epsilon + (epsilon/Qs.shape[0])
        return policy_s

    # INSTRUMENTAL FUNCTIONS GLIE---------------------------------------------
    def _GLIE_map_constant(self):
        return self.conf['GLIE_param']

    def _GLIE_map_dec_bias(self):
        if self.iter_num <= self.convergence_iters:
            epsilon = (((self.conf['GLIE_param'] - 1)/self.convergence_iters)*self.iter_num) + 1
        else:
            epsilon = self.conf['GLIE_param']
        return epsilon

    def _GLIE_map_iter_inverse(self):
        return 1/self.iter_num

    # TEMPORL DIFFERENCE CONTROL METHODS
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # INSTRUMENTAL FUNCTIONS : SARSA------------------------------------------
    def _init_st_sarsa(self, state):
        self.mem['st'] = state
        self.mem['at'] = self._select_action_from_Q(state, self.Q, self.nA, self.epsilon)

    def _select_action_sarsa(self, state):
        # Remember Sarsa chooses at_1 for estimating q
        # So return here precomputed at and compute at_1 in step function
        return self.mem['at']

    def _step_sarsa(self):
        self.mem['at_1'] = self._select_action_from_Q(self.mem['st_1'], self.Q, self.nA, self.epsilon)
        self.Q = self._update_Q_sarsa(self.Q, \
                                 self.conf['alpha'], self.conf['gamma'], \
                                 self.mem['st'], self.mem['at'], self.mem['rt_1'],\
                                 self.mem['st_1'], self.mem['at_1'])
        # Sarsa needs the step explicitly
        self.mem['st'] = self.mem['st_1']
        self.mem['at'] = self.mem['at_1']

    def _update_Q_sarsa(self, Q, alpha, gamma, st, at, rt_1, st_1 = None, at_1 = None):
        if st_1 == None:
            Q[st][at] = Q[st][at] + alpha*(rt_1 - Q[st][at])
        else:
            Q[st][at] = Q[st][at] + alpha*((rt_1 + (gamma*Q[st_1][at_1])) - Q[st][at])
        return Q

    # ------------------------------------------------------------------------
    # INSTRUMENTAL FUNCTIONS : SARSAMAX---------------------------------------
    def _init_st_sarsamax(self, state):
        return None

    def _select_action_sarsamax(self, state):
        self.mem['at'] = self._select_action_from_Q(state, self.Q, self.nA, self.epsilon)
        return self.mem['at']

    def _step_sarsamax(self):
        self.mem['at_1'] = None
        self.Q = self._update_Q_sarsamax(self.Q, self.conf['alpha'], self.conf['gamma'], \
                                         self.mem['st'], self.mem['at'], self.mem['rt_1'],\
                                         self.mem['st_1'])
        # Explicit time step
        self.mem['st'] = self.mem['st_1']

    def _update_Q_sarsamax(self, Q, alpha, gamma, st, at, rt_1, st_1 = None):
        if st_1 == None:
            Q[st][at] = Q[st][at] + alpha*(rt_1 - Q[st][at])
        else:
            Q[st][at] = Q[st][at] + alpha*((rt_1 + (gamma*np.max(Q[st_1]))) - Q[st][at])
        return Q

    # ------------------------------------------------------------------------
    # INSTRUMENTAL FUNCTIONS : EXPECTED SARSA---------------------------------
    def _init_st_expected_sarsa(self, state):
        return None

    def _select_action_expected_sarsa(self, state):
        self.mem['at'] = self._select_action_from_Q(state, self.Q, self.nA, self.epsilon)
        return self.mem['at']

    def _step_expected_sarsa(self):
        self.mem['at_1'] = None
        self.Q = self._update_Q_expected_sarsa(self.Q, self.conf['alpha'], self.conf['gamma'], \
                                             self.mem['st'], self.mem['at'], self.mem['rt_1'],\
                                             self.mem['st_1'])
        # Explicit time step
        self.mem['st'] = self.mem['st_1']

    def _update_Q_expected_sarsa(self, Q, alpha, gamma, st, at, rt_1, st_1 = None):
        if st_1 == None:
            Q[st][at] = Q[st][at] + alpha*(rt_1 - Q[st][at])
        else:
            expected_value = Q[st_1].dot(self._epsilon_greedy_state_probs(Q[st_1], self.epsilon))
            Q[st][at] = Q[st][at] + alpha*((rt_1 + (gamma*expected_value)) - Q[st][at])
        return Q
