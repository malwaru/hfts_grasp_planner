#!/usr/bin/env python


class StochasticOptimizer(object):
    def __init__(self, objective_fn):
        """
            Initializes this stochastic optimizer with the specified objective function.
        """
        self._objective_fn = objective_fn

    def run(self, root, num_iterations):
        """
            Run stochastic optimization on the children of the given node.
            @param root - an object that provides a function get_random_node() which randomly samples
                        a point to evaluate from the optimization domain. The returned point
                        is expected to be evaluable by the objective function provided on initialization
            @param num_iterations - number of iterations to run
        """
        best_node = root.get_random_node()
        best_value = self._objective_fn.evaluate(best_node)
        for i in xrange(num_iterations):
            tmp_node = root.get_random_node()
            o_val = self._objective_fn.evaluate(tmp_node)
            if self._objective_fn.is_better(o_val, best_value):
                best_value = o_val
                best_node = tmp_node
        return best_value, best_node


class StochasticGradientDescent(object):
    def __init__(self, objective_fn):
        """
            Initializes this stochastic gradient descent optimizer with the given objective function.
        """
        self._objective_fn = objective_fn

    def run(self, root, num_iterations):
        """
            Run stochastic gradient descent optimization on the children of the given node.
            It starts at a random child and then locally optimizes on its neighbors
            @param root - an object that provides a function get_random_node() which randomly samples
                         a node to evaluate from the optimization domain. The returned node
                        is expected to be evaluable by the objective function provided on initialization.
                        Similarly, it is expected to have a function get_random_neighbor(node) that returns
                        a random neighbor for that node
            @param num_iterations - number of iterations to run
        """
        best_node = root.get_random_node()
        best_value = self._objective_fn.evaluate(best_node)
        for i in xrange(num_iterations):
            tmp_node = root.get_random_neighbor(best_node)
            o_val = self._objective_fn.evaluate(tmp_node)
            if self._objective_fn.is_better(o_val, best_value):
                best_value = o_val
                best_node = tmp_node
        return best_value, best_node

