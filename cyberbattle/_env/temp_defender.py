# RL Defender

# Implement TODO: (see agent_tabularqlearning.py)
# - Features to create: Feature_reimaged_nodeproperties and Feature_nodes_in_network  X
# - QMatrix (attributes, clear, update, exploit)   X
# - LossEval (all but see template)   X
# - Learner (on_step, exploit, explore)
# - epsilon_greedy_search (based on template)

# NOTE:
# cost of reimaging? How non-RL managing that?
# reward for reimaging node with malware? (intermediary reward not just
#                                          episode reward?)

# NOTE:
# - episode_end is when step_count == 0 (), indicate to reset and
#           record episodic info (see LossEval)
#     - outcome of episode: add line after L1087 in cyberbattle_env.py
#           that gives a reward to defender agent.
# -
# - access Environment (model) and set to class attribute to track?
#               - need on_step function to update it

from os import environ
from sys import ps1
from typing import List, Optional
from cyberbattle.simulation.actions import DefenderAgentActions
import numpy as np
from cyberbattle.simulation.model import Environment

def random_argmax(array):
    """Just like `argmax` but if there are multiple elements with the max
    return a random index instead of returning the first one.
    
    :param array: Numpy array or array-like object.
    """

    max_value = np.max(array)
    max_index = np.where(array == max_value)[0]

    if max_index.shape[0] > 1: # Check if there exist multiple elements with the max.
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    return max_value, max_index

class Feature_reimaged_nodeproperties:
    """
    Feature that is a bitmask indicating node properties present in `nodes` (`nodes` should be
    the last few reimaged nodes).

    :param nodes: List of node IDs.
    :param environment: `Environment` to access nodes from.

    TODO:
    - assert `nodes` list is only a fixed chosen length (3)
    """

    def get_feature_nodes_at(self, environment: Environment, defender_actuator: DefenderAgentActions) -> List:
        """ 
        Obtain NodeIDs of last 3 nodes reimaged (i.e. last 3 nodes malware was detected
        on).
        
        """
        reimaging_node_ids = []

        for node_id in defender_actuator.node_reimaging_progress:
            reimaging_node_ids.append(node_id)

        # Take the last 3 
        # ACCOUNTING FOR FIRST STEP WHERE NO NODES HAVE BEEN REIMAGED
        # But might not have 3 last reimaged notes even after first step in the dict
        # Can't just take whatever is in the dict or history of reimaing (capped) because
        #   that would mean the state_space would be variable? Way to make it not?

        # step uses actions (DefenderAgentActions) - actions.reimage_node(node_id)
        # DefenderAgentActions has attribute: self.node_reimaging_progress: Dict[model.NodeID, int] = dict()
        #       can retrieve last 3 nodes being reimaged (pad if there aren't 3?)
        # Then need to pass in DefenderAgentActions
        pass

    def feature_vector_at(self, environment: Environment, nodes: List) -> np.ndarray:
        """
        Obtain the feature vector for a list of nodes.
        """

        # Create array (nodes x node properties)
        node_prop = np.array([[environment.get_node(nodeID).properties] for nodeID in nodes])
        # Remap to get rid of unknown value 0: 1 -> 1, and -1 -> 0 (and 0-> 0)
        node_prop_remapped = np.int32((1 + node_prop) / 2)

        countby_col = np.sum(node_prop_remapped, axis = 0)

        # Map non-zero elements to 1
        bitmask = (countby_col > 0) * 1

        # NOTE
        # NodeInfo (a comuter node in the enterprise network) class has class
        #           attribute `properties`.
        #           Other potentially relevant attributes:
        #                   - privilege_level: PriviledgeLevel (Access priviledge level on a given node)
        #                   - value: NodeValue (Intrinsic value of a node in [0,11] - can translate into a reward or penalty)
        #                   - sla_weight: (float) Relative node weight used to calculate the cost of stopping this machine/node or its services
        # Note: Environment.get_node(node_id) returns NodeInfo for a node with specified ID

        return bitmask

    def ravel_encode_feature_vector(self, feature_vector) -> int:
        """
        Return the Q Matrix index (ravel) encoding of the feature vector.

        TODO:
        - assert len(feature_vector) is of length property_count. Chain environment this
            code will be tested on has only 3 properties per node. In this case, dim_size
            would then be [2,2,2], where - for each property - there are two options: 
                1. present (1), or
                2. not present (0)
            in the last few reimaged nodes.
        """
        dim_size = [2] * len(feature_vector)
        index: np.int32 = np.ravel_multi_index(feature_vector, dim_size)

        return index

    def flat_size(self):
        """
        TODO: 
        - add dim_size as class attribute. dim_size is currently defined separately in
            both ravel_encode_feature_vector and flat_size.
        """
        dim_size = [2] * 3
        return np.prod(dim_size)

class Feature_nodes_in_network:
    """
    Feature that is the node IDs in the network. (Note: encoding is unnecessary for this
    feature).
    """
    
    def feature_vector_at(self, environment: Environment) -> List:
        nodes = []

        node_generator = environment.nodes()
        for nodeid, node_data in node_generator:
            nodes.append(nodeid)
        
        return nodes

    def flat_size(self):
        """
        TODO:
        - adjust this function so dim_size is not hard-coded.
        """
        dim_size = 10


"""
import numpy as np
lista = [[-1, 1, 1, 0,0], [0, -1, 1, 1,1], [0, 1, 1, 0,-1]]
a = np.array(lista)
a = (a + 1) /2
a
a = np.int32(a)
a
a = np.sum(a, axis = 0)
a
a = (a > 0) *1 # Map non-zero elements to 1
a

prop = [b for b in range(1,4)]
prop
a = np.array(prop)
l = len(a)
l
"""

class QMatrixReimage:

    qm: np.ndarray

    def __init__(self, state_space, action_space, qm: Optional[np.ndarray] = None):
        self.state_space = state_space
        self.action_space = action_space
        self.statedim = state_space.flat_size()
        self.actiondim = action_space.flat_size()
        self.qm = self.clear() if qm is None else qm

        self.last_error = 0

    def shape(self):
        return (self.statedim, self.actiondim)

    def clear(self):
        """Re-initialize the Q-matrix to 0"""
        self.qm = np.zeros(shape=self.shape())
        # self.qm = np.random.rand(*self.shape()) / 100
        return self.qm

    def update(self, current_state: int, action: int, next_state: int, reward, gamma, learning_rate):
        """Update the Q Matrix after taking 'action' in 'current_state'."""

        maxq_atnext, max_index = random_argmax(self.qm[next_state, ]) # select row in Q Matrix corresponding with `next_state`

        # Bellman equation for Q-learning
        temporal_difference = reward + gamma * maxq_atnext - self.qm[current_state, action]
        self.qm[current_state, action] += learning_rate * temporal_difference

        # Loss is calculated using the squared difference between target Q-Value and predicted Q-Value
        square_error = temporal_difference * temporal_difference
        self.last_error = square_error

        return self.qm[current_state, action]

    def exploit(self, state):
        """Exploit the Q-Matrix"""
        expected_q, action = random_argmax(self.qm[state, ])
        return action, expected_q

class LossEval:
    """Loss evaluation for a Q-Learner,
    learner -- The Q learner
    """

    def __init__(self, qmatrix: QMatrixReimage):
        self.qmatrix = qmatrix
        self.this_episode = []
        self.all_episodes = []

    def new_episode(self):
        self.this_episode = []

    def end_of_iteration(self, t, done):
        self.this_episode.append(self.qmatrix.last_error)

    def current_episode_loss(self):
        return np.average(self.this_episode)

    def end_of_episode(self, i_episode, t):
        """Average out the overall loss for this episode"""
        self.all_episodes.append(self.current_episode_loss())



class DefenderQTabularLearner:
    """" doc string"""

    def __init__(self, gamma: float,
                learning_rate: float,
                trained = None):
        
        if trained:
            self.qm = trained.qm # Not QMatrixRimage(qm)?
        else:
            self.qm = QMatrixReimage(Feature_reimaged_nodeproperties, Feature_nodes_in_network)

        self.qm = LossEval(self.qm)
        self.gamma = gamma
        self.learning_rate = learning_rate
    
    
    def on_step(self, environment: Environment, observation, reward, done, info, action_metadata):
        """ Pass `Environment` into this function - like it's passed to the original Defender step()"""
        # Probably need ChosenActionMetadata but adjusted - what adjustments needed?

        # Relationship between model and environment: model.Environment
        # Wherever agenttabularqlearning.py is encoding, its using the 
        # ecoding function in the FeatureEncoder class
        # Therefore, functions needed for first step:
        # - Retrieve observation (this is amounts to just passing in the full environment)
        # - Retrieve features for observation
        # - Encode observation, returning index in Q Matrix.

        # nodes = action_metadata.nodes?
        # nodes would be current state but unravelled?
        feature_vector = self.qm.state_space.feature_vector_at(environment, nodes)
        next_state = self.qm.state_space.ravel_encode_feature_vector(feature_vector)

        self.qm.update(current_state, # current_state from action_metadata
                    action, # action from action_metadata - index on Q Matrix
                    next_state, 
                    reward, # reward is passed in, what is it?
                    self.gamma, 
                    self.learning_rate)

        # 1. Encode the observation to return next_state index in Q Matrix
                # e.g Feature.encode(Environment, nodeIDs)
                # Therefore, nodeIDS needs to be passed in - action_metadata or observation?
        # 2. Update the Q Matrix to reflect changes made by last step
        # 3. Return nothing
        pass
    
    def exploit(self, observation):

        # 1. Encode the observation to current_state index in Q Matrix
        # 2. Exploit Q Matrix using 1. 
        # 3. Return action_metadata and gym_action
        pass

    def explore(self):
        
        # 1. Sample a valid Defender action
        # 2. Return the gym action and ChosenActionMetadata (i.e. return 
        #       same returns as exploit)
        pass