import numpy as np
import pandas as pd
import pandas.testing as pdt
from structure_learning.utils.graph_utils import convert_pkl_graph_to_csv

def test_data_consistency_bet_exps(num_nodes, max_exp, tolerance = 0.000001, base_results_path = "/Volumes/SSD/MCMC"):

    for exp_id in range(0, max_exp):

        print(f"Experiment {exp_id} num_nodes={num_nodes}")

        # load dataset for pi=0.1
        base_data_path = f"{base_results_path}/final_results_noise_0.5_pi_0.1/exp{exp_id}/{num_nodes}_nodes/StructureMCMC/rea0_data.csv"

        # load the data for pi=0.1, to 0.5 and check if the dataset is the same
        base_data = pd.read_csv(base_data_path)
        for pi in [0.1, 0.2, 0.3, 0.4, 0.5]:

            data_path = f"{base_results_path}/final_results_noise_0.5_pi_{pi}/exp{exp_id}/{num_nodes}_nodes/AugmentedMCMC/rea0_data.csv"
            data = pd.read_csv(data_path)

            # assert with threshold 0f 0.000001
            try:
                # Attempt to assert DataFrame equality with a tolerance
                pdt.assert_frame_equal(base_data, data, atol=tolerance, check_dtype=False)
            except AssertionError:
                # If they are not equal within the tolerance, raise an error with your message
                raise AssertionError(f"Data for pi={pi} is not the same as the data for pi=0.1 within a tolerance of {tolerance}")
            print(f"\tpi={pi}\tTest passed")

def test_init_dag_consistency_bet_exps(num_nodes, max_exp, tolerance = 0.000001, base_results_path = "/Volumes/SSD/MCMC"):

    for exp_id in range(0, max_exp):

        print(f"Experiment {exp_id} num_nodes={num_nodes}")

        # load dataset for pi=0.1
        try:
            base_data_path = f"{base_results_path}/final_results_noise_0.5_pi_0.1/exp{exp_id}/{num_nodes}_nodes/StructureMCMC/rea0_initial_DAG.csv"
            base_data = pd.read_csv(base_data_path)
        except FileNotFoundError:
            convert_pkl_graph_to_csv( f"{base_results_path}/final_results_noise_0.5_pi_0.1/exp{exp_id}/{num_nodes}_nodes/StructureMCMC/rea0_initial_DAG.pkl" )

        # load the data for pi=0.1, to 0.5 and check if the dataset is the same

        for pi in [0.1, 0.2, 0.3, 0.4, 0.5]:

            try:
                data_path = f"{base_results_path}/final_results_noise_0.5_pi_{pi}/exp{exp_id}/{num_nodes}_nodes/AugmentedMCMC/rea0_data.csv"
                data = pd.read_csv(data_path)
            except FileNotFoundError:
                convert_pkl_graph_to_csv( f"{base_results_path}/final_results_noise_0.5_pi_0.1/exp{exp_id}/{num_nodes}_nodes/AugmentedMCMC/rea0_initial_DAG.pkl" )


            # assert with threshold 0f 0.000001
            try:
                # Attempt to assert DataFrame equality with a tolerance
                pdt.assert_frame_equal(base_data, data, atol=tolerance, check_dtype=False)
            except AssertionError:
                # If they are not equal within the tolerance, raise an error with your message
                raise AssertionError(f"Data for pi={pi} is not the same as the data for pi=0.1 within a tolerance of {tolerance}")
            print(f"\tpi={pi}\tTest passed")

def test_data_consistency_single_exps(num_nodes, exp_id, max_rea, tolerance = 0.000001, base_results_path = "/Volumes/SSD/MCMC"):

    for id in range(0, max_rea):

        print(f"Experiment {exp_id} num_nodes={num_nodes} realisation {id}")

        # load dataset for pi=0.1
        base_data_path = f"{base_results_path}/final_results_noise_0.5_pi_0.1/exp{exp_id}/{num_nodes}_nodes/StructureMCMC/rea{id}_data.csv"

        # load the data for pi=0.1, to 0.5 and check if the dataset is the same
        base_data = pd.read_csv(base_data_path)
        for pi in [0.1, 0.2, 0.3, 0.4, 0.5]:

            data_path = f"{base_results_path}/final_results_noise_0.5_pi_{pi}/exp{exp_id}/{num_nodes}_nodes/AugmentedMCMC/rea{id}_data.csv"
            data = pd.read_csv(data_path)

            # assert with threshold 0f 0.000001
            try:
                # Attempt to assert DataFrame equality with a tolerance
                pdt.assert_frame_equal(base_data, data, atol=tolerance, check_dtype=False)
            except AssertionError:
                print("Structure vs Augmented MCMC")
                # If they are not equal within the tolerance, raise an error with your message
                raise AssertionError(f"Data for pi={pi} is not the same as the data for pi=0.1 within a tolerance of {tolerance}")
        print(f"\tAugmented MCMC Test passed")

        pi = 0.1


        if num_nodes < 15:
            try:
                data_path = f"{base_results_path}/final_results_noise_0.5_pi_{pi}/exp{exp_id}/{num_nodes}_nodes/PartitionMCMC/original/rea{id}_data.csv"
                data = pd.read_csv(data_path)
            except FileNotFoundError:
                print(f"\nERROR Partition MCMC File {data_path} does not exist\n" )

            try:
                pdt.assert_frame_equal(base_data, data, atol=tolerance, check_dtype=False)
            except AssertionError:
                print("Structure vs Partition MCMC")
                raise AssertionError(f"[ERROR] Data for Partition MCMC for realisation {id} failed within a tolerance of {tolerance}")
            print(f"\tPartition MCMC Test passed")

        try:
            data_path = f"{base_results_path}/final_results_noise_0.5_pi_{pi}/exp{exp_id}/{num_nodes}_nodes/PartitionMCMC/bidag/rea{id}_data.csv"
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"\nERROR BiDAG MCMC File {data_path} does not exist\n" )

        try:
            pdt.assert_frame_equal(base_data, data, atol=tolerance, check_dtype=False)
        except AssertionError:
            print("Structure vs BIDAG MCMC")
            raise AssertionError(f"[ERROR] Data for BIDAG MCMC for realisation {id} failed within a tolerance of {tolerance}")
        print(f"\tPartition MCMC BiDAG Test passed")

        try:
            data_path = f"{base_results_path}/final_results_noise_0.5_pi_{pi}/exp{exp_id}/{num_nodes}_nodes/OrderMCMC/rea{id}_data.csv"
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"\nERROR Order MCMC File {data_path} does not exist\n" )

        try:
            pdt.assert_frame_equal(base_data, data, atol=tolerance, check_dtype=False)
        except AssertionError:
            print("Structure vs Order MCMC")
            raise AssertionError(f"[ERROR] Data for Order MCMC for realisation {id} failed within a tolerance of {tolerance}")
        print(f"\tOrder MCMC Test passed")
