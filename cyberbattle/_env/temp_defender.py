# RL Defender

## NOTES
# - episode_end is when step_count == 0 (), indicate to reset and
#           record episodic info (see LossEval)
#     - outcome of episode: add line after L1087 in cyberbattle_env.py
#           that gives a reward to defender agent.
# - 
# - access Environment (model) and set to class attribute to track?
#               - need on_step function to update it

# Implement TODO: (see agent_tabularqlearning.py)
# - Features to create: nodes_not_reimaging and nodes_running (action_space feature),
#                       here state_space)
# - QMatrix (attributes, clear, update, exploit)
# - LossEval (all but see template) - merge into Learner?
# - Learner (on_step, exploit, explore)
# - epsilon_greedy_search (based on template)



# cost of reimaging? How non-RL managing that?
# reward for reimaging node with malware? (intermediary reward not just
# #                                         episode reward?)
import numpy as np

#DefenderFeature

class QMatrixReimage:

    qm: np.ndarray

    def __init__(self, state_space, action_space, qm):
        self.state_space = state_space
        self.action_space = action_space
        self.statedim = #
        self.actiondim = #
        self.qm = self.clear() if qm is None else qm

        self.last_error = 0

        def shape(self):
            return (self.statedim, self.actiondim)

        def clear(self):
            """Re-initialize the Q-matrix to 0"""
            self.qm = np.zeros(shape=self.shape())
            # self.qm = np.random.rand(*self.shape()) / 100
            return self.qm
