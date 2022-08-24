    def feature_vector_at(self, environment: Environment, nodes_bitmask: List) -> np.ndarray:
        """
        Obtain the feature vector for a list of nodes.
        """

        # Create array (nodes x node properties)
        node_prop = np.array([[environment.get_node(nodeID).properties] for nodeID in nodes])
        # Remap to get rid of unknown value 0: 1 -> 1, and -1 -> 0 (and 0-> 0)
        node_prop_remapped = np.int32((1 + node_prop) / 2)

        countby_col = np.sum(node_prop_remapped, axis=0)

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
