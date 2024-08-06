"""

"""
import random
import numpy as np
from mcmc.utils.graph_utils import collect_node_scores, compare_graphs, index_to_node_label, initial_graph_pc, generate_DAG
from mcmc.proposals import StructureLearningProposal, GraphProposal
from mcmc.scores import Score, BGeScore
from mcmc.mcmc import MCMC

class StructureMCMC(MCMC):
    """
    Implementation of Structure MCMC.
    """
    def __init__(self, initial_graph : np.ndarray = None, max_iter : int = 30000, proposal_object : StructureLearningProposal = None, score_object : Score = None, data = None, pc_init = True):
        """
        Initilialise Structure MCMC instance.

        Parameters:
            initial_graph (numpy.ndarray | None): Initial graph for the MCMC simulation. If None, simulation starts with a graph with no edges.
            max_iter (int): The number of MCMC iterations to run.
            proposal_object (StructureLearningProposal): A proposal object.
            score_object (Score): A score object implementing compute().
        """
        if initial_graph is None:
            if pc_init:
                print('Running PC algorithm')
                if score_object is None and data is None:
                    raise Exception("Data must be provided.")
                initial_graph = initial_graph_pc(score_object.data if score_object else data)
            else: # start with random
                n_nodes = len(score_object.data.columns)
                initial_graph = generate_DAG(n_nodes, 0.5)

        if score_object is None:
            if data is None:
                raise Exception("Data must be provided.")
            else:
                score_object = BGeScore(data=data, incidence=initial_graph)
        elif type(score_object) == str:
            if score_object.lower() == 'bge':
                if data is None:
                    raise Exception("Data must be provided")
                else:
                    score_object = BGeScore(data=data, incidence=initial_graph)
            else:
                raise Exception(f"Unsupported score {score_object}")

        if proposal_object is None:
            proposal_object = GraphProposal(initial_graph)

        super().__init__(score_object.data, initial_graph, max_iter, score_object, proposal_object)

        self.to_string = f"Struct_MCMC_n_{self._num_nodes}_iter_{self._max_iter}"

    def __str__(self):
        return self.to_string

    # main MCMC function that needs to be implemented
    def log_acceptance_ratio(self, posterior_Gcurr : float, posterior_Gprop : float,
                             Q_Gcurr_Gprop : float, Q_Gprop_Gcurr : float):
        """
        Calculate log acceptance ratio.

        Parameters:
            posterior_Gcurr (float):
            posterior_Gprop (float):
            Q_Gcurr_Gprop (flaot):
            Q_Gprop_Gcurr (float):

        Returns:
            (float)
        """

        try:
            numerator = posterior_Gprop + np.log(Q_Gcurr_Gprop)
        except:
            print("RuntimeWarning: divide by zero encountered in log")
            print(f"\tposterior_Gcurr: {posterior_Gcurr}")
            print(f"\tQ_Gcurr_Gprop: {Q_Gcurr_Gprop}")
            Q_Gcurr_Gprop = 0.000001

        try:
            denominator = posterior_Gcurr + np.log(Q_Gprop_Gcurr)
        except:
            print("RuntimeWarning: divide by zero encountered in log")
            print(f"\tposterior_Gprop: {posterior_Gprop}")
            print(f"\tQ_Gprop_Gcurr: {Q_Gprop_Gcurr}")
            Q_Gprop_Gcurr = 0.000001

        return  min(0, numerator - denominator)

    def acceptance_ratio(self, posterior_Gcurr : float, posterior_Gprop : float,
                         Q_Gcurr_Gprop : float, Q_Gprop_Gcurr : float):
        """
        Calculate acceptance ratio.

        Parameters:
            posterior_Gcurr (float):
            posterior_Gprop (float):
            Q_Gcurr_Gprop (flaot):
            Q_Gprop_Gcurr (float):

        Returns:
            (float)
        """
        numerator = posterior_Gprop * Q_Gcurr_Gprop
        denominator = posterior_Gcurr * Q_Gprop_Gcurr

        return min(1, numerator/denominator)

    def run(self):
        """
        Run MCMC simulation.

        Returns:
            (list (dict)): information on graph samples
            (float): acceptance rate
        """
        mcmc_res = {}
        iter_indx = 0
        ACCEPT = 0

        # start with the initial current graph:
        # compute the score of the graph given the data
        G_curr = self.initial_graph.copy()
        G_curr_operation = "initial"

        # compute the score for the initial graph
        score_dict = self.score_object.compute()
        score_Gcurr = score_dict['score']

        mcmc_res[0] = {"graph": G_curr,
                        "score": score_Gcurr,
                        "operation":G_curr_operation,
                        "accepted" : 0,
                        "Q_Gprop_Gcurr" : 1,
                        "Q_Gcurr_Gprop" : 1,
                        "score_Gprop" : 1,
                        "acceptance_prob" : 0}

        node_score_Gcurr = collect_node_scores(score_dict)

        for _ in range(self.max_iter):

            accept_indx = 0
            acceptance_prob = 0

            # propose a new graph and compute the proposal distribution Q
            self.proposal_object.current_graph = G_curr
            G_prop, G_prop_operation = self.proposal_object.propose_DAG()

            # compute the proposal distribution Q
            Q_Gcurr_Gprop = self.proposal_object.prob_Gcurr_Gprop
            Q_Gprop_Gcurr = self.proposal_object.prob_Gprop_Gcurr

            if G_prop_operation == "stay_still":

                mcmc_res[iter_indx] = {"graph": self.proposal_object.current_graph,
                                    "score": score_Gcurr,
                                    "operation": G_prop_operation,
                                    "accepted" : 0,
                                    "Q_Gprop_Gcurr" : Q_Gprop_Gcurr,
                                    "Q_Gcurr_Gprop" : Q_Gcurr_Gprop,
                                    "score_Gprop" : 0,
                                    "acceptance_prob" : 0}

                iter_indx += 1
                continue

            # we need to update the graph so we can extract the parents of the node
            self.score_object.incidence = G_prop
            node_score_Gprop = node_score_Gcurr.copy()

            rescored_nodes = compare_graphs(G_curr, G_prop, G_prop_operation, index_to_node_label(self.node_labels))

            if G_prop_operation[0] in set(["add_edge", "delete_edge"]):
                node_score_Gprop[rescored_nodes] = self.score_object.compute_node(rescored_nodes)['score']
            else:
                node_score_Gprop[rescored_nodes[0]] = self.score_object.compute_node(rescored_nodes[0])['score']
                node_score_Gprop[rescored_nodes[1]] = self.score_object.compute_node(rescored_nodes[1])['score']

            score_Gprop = sum(list(node_score_Gprop.values()))

            if self.score_object.is_log_space:
                acceptance_prob = self.log_acceptance_ratio(score_Gcurr, score_Gprop, Q_Gcurr_Gprop, Q_Gprop_Gcurr)
                u = np.log(np.random.uniform(0, 1))
            else:
                acceptance_prob = self.acceptance_ratio(score_Gcurr, score_Gprop, Q_Gcurr_Gprop, Q_Gprop_Gcurr)
                u =  random.uniform(0,1)

            if u < acceptance_prob:

                ACCEPT += 1
                G_curr = G_prop
                self.proposal_object.current_graph = G_prop
                score_Gcurr = score_Gprop
                G_curr_operation = G_prop_operation
                node_score_Gcurr = node_score_Gprop.copy()
                accept_indx = 1


            mcmc_res[iter_indx] = {"graph": self.proposal_object.current_graph,
                                    "score": score_Gcurr,
                                    "operation": G_curr_operation,
                                    "accepted" : accept_indx,
                                    "Q_Gprop_Gcurr" : Q_Gprop_Gcurr,
                                    "Q_Gcurr_Gprop" : Q_Gcurr_Gprop,
                                    "score_Gprop" : score_Gprop,
                                    "score_Gcurr" : score_Gcurr,
                                    "acceptance_prob" : acceptance_prob}
            # reset index
            accept_indx = 0
            iter_indx += 1

        return mcmc_res, np.round(ACCEPT/self.max_iter, 4)


    def get_mcmc_res_graphs(self, results):
        """
        Returns list of sampled graphs from MCMC simulation results.

        Parameters:
            results (dict): MCMC simulation results.

        Returns:
            (list): sampled graphs
        """
        return [result['graph'] for _,(i,result) in enumerate(results.items())]

    def get_mcmc_res_operations(self, results):
        """
        Returns sequence of operations from MCMC simulation results.

        Parameters:
            results (dict): MCMC simulation results.

        Returns:
            (list): operations that generated the sampled graphs
        """
        return [result['operation'] for _,(i,result) in enumerate(results.items())]

    def get_mcmc_res_scores(self, results):
        """
        Returns computed scores for each sampled graph from the simulation results.

        Parameters:
            Parameters:
            results (dict): MCMC simulation results.

        Returns:
            (list): graph scores
        """
        return [result['score'] for _,(i,result) in enumerate(results.items())]

    def get_mcmc_res_accepted_graphs(self, results):
        """
        Filters accepted graphs from the simulation results.

        Parameters:
            Parameters:
            results (dict): MCMC simulation results.

        Returns:
            (list): accepted sampled graphs
        """
        mcmc_accepted_graph_lst = []
        mcmc_accepted_graph_indx = []
        for _,(i,result) in enumerate(results.items()):
            mcmc_accepted_graph_indx.append(result['accepted'])
            if result['accepted'] == 1:
                mcmc_accepted_graph_lst.append(result['graph'])
        return mcmc_accepted_graph_lst, mcmc_accepted_graph_indx
