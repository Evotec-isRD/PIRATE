# Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import argparse
import pickle
import subprocess
import time
import copy
import logging
import warnings
import os

# suppress warnings when loading things
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)
warnings.filterwarnings("ignore")

# basic logger for script
logger = logging
logger.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d--%H:%M:%S"
)


class PirateEvolutionTool:
    """Deletion Tool for protein construct design"""

    def __init__(self, sequence=None, max_residues_to_alter=None, max_iterations=None, protected_residues=None,
                 residues_to_modify=None, gamma=5, epsilon=5, selection_criterion="disorder") -> None:
        """init function
        :param sequence: A string specifying the query sequence
        :return: None"""

        self.sequence = str(sequence)
        if max_residues_to_alter is not None:
            self.max_residues_to_alter = int(max_residues_to_alter)
        if not max_residues_to_alter:
            self.max_residues_to_alter = int(len(self.sequence))
        if max_iterations is not None or 0:
            self.max_iterations = max_iterations
        if not max_iterations or max_iterations == 0:
            self.max_iterations = 999999999999999
        self.evolution_counter = 0
        # Here we check if None or == "None" due to the difference in how the arguments will be
        # if the user is running this from the command line or from the Streamlit app
        if residues_to_modify and residues_to_modify != "None":
            self.residues_to_modify = residues_to_modify.split(",")
            self.residues_to_modify = [int(float(e)) for e in self.residues_to_modify]
        if not residues_to_modify or residues_to_modify == "None":
            self.residues_to_modify = []
        if protected_residues and protected_residues != "None":
            self.protected_residues = protected_residues.split(",")
            self.protected_residues = [int(e) for e in self.protected_residues]
        if not protected_residues or protected_residues == "None":
            self.protected_residues = []
        self.starting_sequence = str(sequence)
        self.starting_plddt = 0.0
        self.starting_disorder = 1.0
        self.gamma = int(gamma)  # number of mutations to check per residue
        self.epsilon = int(epsilon)  # number of residues to consider per evolution
        self.afsm1_model = ""
        self.afsm2_model = ""
        self.afsm3_model = ""
        self.pirate_model = ""
        self.current_sequence = self.sequence
        self.best_plddt_sequence = self.sequence  # sequence with best plddt score
        self.best_disorder_sequence = self.sequence  # sequence with the least mean disorder
        self.current_plddt = 0.0
        self.current_disorder = 1.0
        self.best_plddt = 0.0
        self.best_disorder = 1.0
        self.neighborhood_size = int(3)
        self.mutation_dict = {}  # dictionary holding current mutations
        self.best_plddt_mutation_dict = {}  # mutations leading to best plddt score
        self.best_disorder_mutation_dict = {}  # mutations leading to best reduction in disorder
        self.num_changes_list = []
        self.sequences_list = []
        self.mutation_dicts_list = []
        self.delta_disorder_list = []
        self.delta_plddt_list = []

        self.mutation_list = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F",
                              "P", "S", "T", "W", "Y", "V"]
        self.selection_criterion = selection_criterion

    def load_models(self) -> None:
        """Loads models for prediction"""
        local_path = pathlib.Path().absolute()
        model_path = str(local_path.parents[0]) + "/models/"
        afsm1_path = model_path + "afsm1"
        afsm2_path = model_path + "afsm2"
        afsm3_path = model_path + "afsm3"
        pirate_path = model_path + "pirate.pkl"

        self.afsm1_model = tf.keras.models.load_model(afsm1_path, custom_objects=None, compile=True, options=None)
        logger.info("afsm1 loaded")
        self.afsm2_model = tf.keras.models.load_model(afsm2_path, custom_objects=None, compile=True, options=None)
        logger.info("afms2 loaded")
        self.afsm3_model = tf.keras.models.load_model(afsm3_path, custom_objects=None, compile=True, options=None)
        logger.info("afsm3 loaded")
        self.pirate_model = pickle.load(open(pirate_path, 'rb'))
        logger.info("pirate loaded")

    @staticmethod
    def encode_data(data: str, input_size: int) -> (np.ndarray, int, int):
        """
        This function encodes and pads data with mirror images for afsm1/2 prediction.

        :param data: A string representing raw sequence to predict
        :param input_size: An integer specifying window size. This should usually be 4096 for afsm1/2 models.
        :return fasta: Numerically encoded and padded sequence as numpy array
        :return split: Index location where N-term padding ends (integer)
        :return stop_pos: Index location where C-term padding starts (integer)
        """
        residue_dictionary = {"A": 1, "E": 2, "L": 3, "M": 4, "C": 5, "D": 6, "F": 7, "G": 8,
                              "H": 9, "K": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15,
                              "W": 16, "Y": 17, "T": 18, "V": 19, "I": 20}

        fasta = list(str(data))
        # Encode data
        for index, value in enumerate(fasta):
            fasta[index] = residue_dictionary[value]
        # Pad data

        # Invert FASTA and make list 200 times the length to avoid edge cases where FASTA is small
        padding = fasta[::-1] * 2000

        split = int((input_size - len(fasta)) / 2)
        last_padding_len = input_size - len(fasta) - split

        stop_pos = int(split + len(fasta))
        padding_1 = padding[-split:]
        padding_2 = padding[:last_padding_len]
        fasta = padding_1 + fasta + padding_2

        # Reshape data for input
        fasta = np.array(fasta).reshape(-1, input_size, 1)
        # Normalize data by subtracting training mean and dividing by training std. deviation
        fasta = (fasta - 10.108613363425793) / 6.034641898334733
        return fasta, split, stop_pos

    @staticmethod
    def encode_afsm3_data(data, input_size) -> (np.ndarray, int, int):
        """
        This function encodes and pads data with mirror images for afsm3 prediction.

        :param data: A string representing raw sequence to predict
        :param input_size: An integer specifying window size. This should usually be 2048 for afsm3.
        :return fasta: Numerically encoded and padded sequence as numpy array
        :return split: Index location where N-term padding ends (integer)
        :return stop_pos: Index location where C-term padding starts (integer)
        """
        residue_dictionary = {"A": 1, "E": 2, "L": 3, "M": 4, "C": 5, "D": 6, "F": 7, "G": 8,
                              "H": 9, "K": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15,
                              "W": 16, "Y": 17, "T": 18, "V": 19, "I": 20}

        fasta = list(str(data))
        # Encode data
        for index, value in enumerate(fasta):
            fasta[index] = residue_dictionary[value]
        # Pad data

        # Invert FASTA and make list 200 times the length to avoid edge cases where FASTA is small
        padding = fasta[::-1] * 2000

        split = int((input_size - len(fasta)) / 2)
        last_padding_len = input_size - len(fasta) - split

        stop_pos = int(split + len(fasta))
        padding_1 = padding[-split:]
        padding_2 = padding[:last_padding_len]
        fasta = padding_1 + fasta + padding_2

        # Reshape data for input
        fasta = np.array(fasta).reshape(-1, input_size, 1)
        # Normalize data by subtracting training mean and dividing by training std. deviation
        fasta = (fasta - 10.15) / 5.98
        return fasta, split, stop_pos

    def predict_data(self, fasta: str, model: object, input_size: int) -> list:
        """
        Generate afsm1/2 prediction for sequence.

        :param fasta: raw sequence encoded as a string
        :param model: afsm1/2 model (object)
        :param input_size: integer specifying window size for afsm1/2 models (should be 4096)

        :return: A list of predictions (will either be predicted MAE or pLDDT)
        """

        data, start_pos, stop_pos = self.encode_data(fasta, input_size)
        prediction = model.predict(data).reshape(input_size, 1)
        prediction = prediction[start_pos:stop_pos]
        prediction = [float(i) for i in prediction]

        return prediction

    def afsm3_data(self, fasta: str, model: object, input_size: int) -> list:
        """
        Generate afsm3 prediction for sequence. Will return a list of
        probabilities representing the probability of each residue crystallizing

        :param fasta: raw sequence encoded as a string
        :param model: afsm3 model (object)
        :param input_size: integer specifying window size for afsm3 model (should be 2048)

        :return: A list of predictions (probability of crystallization)
        """

        data, start_pos, stop_pos = self.encode_afsm3_data(fasta, input_size)
        prediction = model.predict(data)[0]
        prediction = list(prediction[:, 1])
        prediction = prediction[start_pos:stop_pos]
        prediction = [float(i) for i in prediction]

        return prediction

    @staticmethod
    def encode_sequence(fasta: str) -> list:
        """
        Encode raw sequence as ordinals
        :param fasta: a string representing raw sequence
        :return: a list of ordinal-encoded residues
        """
        residue_dictionary = {"A": 1, "E": 2, "L": 3, "M": 4, "C": 5, "D": 6, "F": 7, "G": 8,
                              "H": 9, "K": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15,
                              "W": 16, "Y": 17, "T": 18, "V": 19, "I": 20}

        fasta = list(str(fasta))
        # Encode data
        for index, value in enumerate(fasta):
            fasta[index] = int(residue_dictionary[value])

        return fasta

    def encoder_predictions(self, sequence: str) -> [list, list, list, list]:
        """
        Using afsm1, afsm2, afsm3 and ordinal encoding, encode input sequence

        :param sequence: protein sequence formatted as a string
        :return: four lists (mean alignment error, plddt, afsm3 predictions, and ordinals)
        """

        afsm1_pred = self.predict_data(sequence, self.afsm1_model, 4096)
        afsm2_pred = list(np.array(self.predict_data(sequence, self.afsm2_model, 4096)) / 100.0)
        afsm3_pred = self.afsm3_data(sequence, self.afsm3_model, 2048)
        ordinal_list = self.encode_sequence(sequence)

        return afsm1_pred, afsm2_pred, afsm3_pred, ordinal_list

    def predict_disorder(self, sequence, afsm1_pred, afsm2_pred, afsm3_pred, ordinal_list):
        """
        predict the per-residue disorder of the input sequence using PIRATE methodology.
        :param sequence: string representing raw protein sequence
        :param afsm1_pred: asfm1 mae predictions
        :param afsm2_pred: afsm2 plddt predictions
        :param afsm3_pred: afsm3 predictions
        :param ordinal_list: encoded ordinal values
        :return: list of disorder predictions where each value is a per-residue probability of disorder
        """
        predictions = []

        win_size = 11

        start, label, stop = 0, int(win_size), int((win_size * 2) + 1)

        while stop < len(sequence) + 1:
            prediction = self.pirate_model.predict_proba(
                afsm1_pred[start:stop] + afsm2_pred[start:stop] + afsm3_pred[start:stop] + ordinal_list[start:stop])[0]
            predictions.append(prediction)

            start += 1
            label += 1
            stop += 1

        predictions = [0] * win_size + predictions

        predictions += [0] * win_size

        return predictions

    def rank_disorder(self, sequence) -> list:
        """
        Generate order/disorder predictions for each residue in self.sequence.
        :param sequence: The raw protein sequence to predict
        :return: a list of predictions where the residue indexes are ranked by disorder (descending order)
        """
        # list to hold predictions
        predictions = []
        # window size of predictions
        win_size = 11

        afsm1_pred, afsm2_pred, afsm3_pred, ordinal_list = self.encoder_predictions(sequence)

        start, label, stop = 0, int(win_size), int((win_size * 2) + 1)
        while stop < len(sequence) + 1:
            prediction = self.pirate_model.predict_proba(
                afsm1_pred[start:stop] + afsm2_pred[start:stop] + afsm3_pred[start:stop] +
                ordinal_list[start:stop])[0]
            predictions.append(prediction)

            start += 1
            label += 1
            stop += 1

        predictions = [0] * win_size + predictions + [0] * win_size

        idxs = np.argsort(predictions)[::-1].tolist()

        return idxs

    def disorder_probability(self, sequence: str, location: int or str) -> float:
        """
        Generate mean probability of disorder for a region of a sequence.
        :param sequence: raw sequence formatted as a string
        :param location: the location where the mutation was made. This is used to focus probability prediction
        or indicate to predict mean probability for whole sequence
        :return: a mean disorder probability for the sequence
        """
        # remove dashes (holders for deletions)
        sequence = sequence.replace("-", "")
        # list to hold predictions
        predictions = []
        # generate encodings for sequence
        afsm1_pred, afsm2_pred, afsm3_pred, ordinal_list = self.encoder_predictions(sequence)
        # window size of predictions
        win_size = 11

        start, label, stop = 0, int(win_size), int((win_size * 2) + 1)

        while stop < len(sequence) + 1:
            prediction = self.pirate_model.predict_proba(
                afsm1_pred[start:stop] + afsm2_pred[start:stop] + afsm3_pred[start:stop] +
                ordinal_list[start:stop])[0]
            predictions.append(prediction)

            start += 1
            label += 1
            stop += 1

        mean_proba = 1.0
        if location != "all":
            # Calculate disorder probability of window at location. Add windows to predictions so indexing is correct.
            predictions = [0.0] * win_size + predictions
            predictions = predictions + [0.0] * win_size
            mean_proba = float(np.mean(np.array(predictions[location - win_size:location + win_size + 1])))
        if location == "all":
            mean_proba = float(np.mean(np.array(predictions)))

        return mean_proba

    @staticmethod
    def fold_sequence(sequence: str) -> (str or None):
        """
        Generate PDB for sequence using ESMFold API

        :param sequence: a query sequence formatted as a string
        :return: a PDB structure formatted as a string
        """
        model = ""

        if len(sequence) > 400:
            logger.info("sequence too long to use API")
            return

        while len(model) < 100:
            command = ["curl", "-X", "POST", "--data",
                       str(sequence),
                       "https://api.esmatlas.com/foldSequence/v1/pdb/"]

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            model = result.stdout

            time.sleep(5)

        return model

    @staticmethod
    def get_model_plddt(model):

        model_list = model.split("\n")
        plddt_list = []
        for count, e in enumerate(model_list):
            if count > 21:
                plddt_list.append(float(e[e.index("1.00  ") + 6:e.index("1.00  ") + 10]))

        plddt_array = np.array(plddt_list)

        return float(np.mean(plddt_array) + np.min(plddt_array)) / 2.0

    def protect_residues(self, disorder_ranking):
        """
        Updates disorder predictions to make any 'protected' residue's
        classification as ordered.
        :param disorder_ranking: A list of disorder rankings (by residue index)
        :return: An updated list of disorder predictions where all protected residues are classified
        as ordered
        """

        idxs = list(range(len(disorder_ranking)))
        eleven_terminal = idxs[:10] + idxs[-10:]
        # remove terminal 11 residues from disorder ranking
        for count, res_idx in reversed(list(enumerate(disorder_ranking))):
            if int(res_idx) in eleven_terminal:
                del disorder_ranking[count]

        # keep only residues specified in residues to modify (if any are specified)
        if len(self.residues_to_modify) > 0:
            # adjust residue locations to account for deletions
            temp_modify_residues = self.residues_to_modify
            for count, residue in enumerate(temp_modify_residues):
                deletion_counter = 0
                for key, value in self.mutation_dict.items():
                    if key + 1 < residue and value == "-":
                        deletion_counter += 1

                temp_modify_residues[count] = int(residue) - int(deletion_counter)
            # delete residues not specified in residues to modify if list contains elements
            for count, res_idx in reversed(list(enumerate(disorder_ranking))):
                if int(res_idx + 1) not in temp_modify_residues:
                    del disorder_ranking[count]

        temp_protected_residues = self.protected_residues
        # adjust residue locations according to deletions
        for count, residue in enumerate(temp_protected_residues):
            deletion_counter = 0
            for key, value in self.mutation_dict.items():
                if key + 1 < residue and value == "-":
                    deletion_counter += 1

            temp_protected_residues[count] = int(residue) - int(deletion_counter)

        # delete protected residues in reverse order
        for count, res_idx in reversed(list(enumerate(disorder_ranking))):

            if int(res_idx + 1) in temp_protected_residues:
                del disorder_ranking[count]

        return disorder_ranking

    def disorder_score_mutations(self, residue_idx: int) -> list:
        """
        Calculate the change in regional disorder for all possible mutations at specified index.
        Mutations that cause predicted disorder to decrease in the region are returned as a list.

        :param residue_idx: location to evaluate mutations
        :return: A list of approved mutations

        """
        # get region disorder score for current sequence
        ref_score = self.disorder_probability(self.current_sequence, residue_idx)
        temp_scores = []  # list to hold mutation scores for site
        approved_mutations = []
        for mutation in self.mutation_list:
            temp_sequence = self.current_sequence[:residue_idx] + mutation + self.current_sequence[residue_idx + 1:]
            temp_score = self.disorder_probability(temp_sequence, residue_idx)
            # a good score should decrease disorder compared to the current sequence
            temp_score = ref_score - temp_score
            temp_scores.append(temp_score)

        # make list of mutations where disorder is lower than current sequence
        for counter, temp_score in enumerate(temp_scores):
            if temp_score > 0:
                approved_mutations.append(self.mutation_list[counter])

        # Only keep the top disorder-reducing mutations as specified by gamma parameter
        if len(approved_mutations) > self.gamma:
            idx = np.argsort(np.array(temp_scores))[::-1].tolist()
            approved_mutations = []
            for i in range(self.gamma):
                approved_mutations.append(self.mutation_list[idx[i]])

        return approved_mutations

    def get_best_plddt(self, residue_idx: int, approved_mutations: list) -> (str, float):
        """
        Given an index to perform mutations and a list of approved mutations, find
        the mutation that gives the best plddt score by making these mutations in the
        current_sequence.

        :param residue_idx: an integer specifying the location index to make mutations
        :param approved_mutations: a list of mutations that are predicted to decrease disorder at that locus
        :return: the best mutation (string) and the corresponding plddt score (float)
        """
        plddt_scores = []
        subtraction_factor = None
        ref_sequence = ""
        original_idx = residue_idx

        if len(self.current_sequence) <= 400:
            ref_sequence = self.current_sequence

        if len(self.current_sequence) > 400:
            if residue_idx <= 200:
                ref_sequence = self.current_sequence[:400]
            else:
                subtraction_factor = int(residue_idx - 200)
                residue_idx = 200
                ref_sequence = self.current_sequence[subtraction_factor:subtraction_factor + 400]

        ref_plddt = self.get_model_plddt(self.fold_sequence(ref_sequence))

        logger.info(f"Reference pLDDT for site {original_idx + 1}: {ref_plddt}")

        for mutation in approved_mutations:
            # mutate sequence
            temp_sequence = ref_sequence[:residue_idx] + mutation + \
                            ref_sequence[residue_idx + 1:]
            # fold sequence
            temp_model = self.fold_sequence(temp_sequence.replace("-", ""))
            # calculate plddt
            plddt = self.get_model_plddt(temp_model)
            plddt -= ref_plddt  # net improvement in plddt

            plddt_scores.append(plddt)

            if subtraction_factor:
                logger.info(f"residue: {str(residue_idx + 1 + subtraction_factor)}, mutation: {str(mutation)}, "
                            f"plddt score: {str(plddt)}")
            if not subtraction_factor:
                logger.info(f"residue: {str(residue_idx + 1)}, mutation: {str(mutation)}, plddt score: {str(plddt)}")

        best_mutation = approved_mutations[np.argmax(np.array(plddt_scores))]
        best_plddt = np.max(np.array(plddt_scores))

        return best_mutation, best_plddt

    def get_best_disorder(self, residue_idx: int, approved_mutations: list) -> (str, float):
        """
        Given an index to perform mutations and a list of approved mutations, find
        the mutation that gives the best reduction in global disorder by making these mutations in the
        current_sequence.

        :param residue_idx: an integer specifying the location index to make mutations
        :param approved_mutations: a list of mutations that are predicted to decrease disorder at that locus
        :return: the best mutation (string) and the corresponding plddt score (float)
        """

        global_disorder_scores = []

        for mutation in approved_mutations:
            # mutate sequence
            temp_sequence = self.current_sequence[:residue_idx] + mutation + \
                            self.current_sequence[residue_idx + 1:]
            # get global disorder
            global_disorder = self.disorder_probability(temp_sequence, "all")

            # append score for mutation
            global_disorder_scores.append(global_disorder)
            logger.info(f"residue: {str(residue_idx + 1)}, mutation: {str(mutation)}, "
                        f"global disorder: {str(global_disorder)}")

        best_mutation = approved_mutations[np.argmin(np.array(global_disorder_scores))]
        best_plddt = np.min(np.array(global_disorder_scores))

        return best_mutation, best_plddt

    def mutate_residues(self, disorder_ranking: list) -> str:
        """
        This function automates the process of testing mutations/deletions at ten residues where
        disorder is predicted to be highest in the sequence. First, we check for any previously identified "best"
        mutations. If these have already been computed, we swap these into the sequence. Then, at any
        new sites of predicted disorder we test each canonical residue as well as a deletion ('-').
        Whatever the best result, we add that to the self.mutation_dict dictionary and use that going
        forward. The best sequence is returned after all residues have been predicted.

        :param disorder_ranking: A list of predictions where residue indexes are ranked in descending
        order by predicted disorder
        :return: The best sequence formatted as a string
        """

        # iterate through predictions
        best_site_scores = []  # list to hold best result at each site
        best_site_mutations = []  # list to hold the best mutation at each site
        mutation_sites = []  # list to track mutation sites where a mutation could be made

        disorder_ranking = disorder_ranking[:self.epsilon]

        for count, residue_idx in enumerate(disorder_ranking):
            # get mutations that will decrease disorder at site
            approved_mutations = self.disorder_score_mutations(residue_idx)
            # in none exist, skip to next site
            if len(approved_mutations) < 1:
                continue

            if self.selection_criterion == "plddt":
                best_mutation, best_score = self.get_best_plddt(residue_idx, approved_mutations)

            else:
                best_mutation, best_score = self.get_best_disorder(residue_idx, approved_mutations)
            mutation_sites.append(residue_idx)  # keep track of residue index manually
            best_site_scores.append(best_score)  # save best score for each site
            best_site_mutations.append(best_mutation)  # best mutation for each site

        # if no valid mutations can be made given the current constraints, return without
        # updating the mutation dictionary or current sequence
        if len(best_site_mutations) < 1:
            return self.current_sequence

        if self.selection_criterion == "plddt":
            overall_best_mutation = best_site_mutations[np.argmax(np.array(best_site_scores))]
            mutation_site = mutation_sites[np.argmax(np.array(best_site_scores))]

        else:
            overall_best_mutation = best_site_mutations[np.argmin(np.array(best_site_scores))]
            mutation_site = mutation_sites[np.argmin(np.array(best_site_scores))]

        deletion_counter = 0
        for key, value in self.mutation_dict.items():
            if key < mutation_site and value == "-":
                deletion_counter += 1

        self.mutation_dict[mutation_site + deletion_counter + 1] = overall_best_mutation

        output_sequence = (self.current_sequence[:mutation_site] + overall_best_mutation +
                           self.current_sequence[mutation_site + 1:])

        # remove dashes (place holders for deletions)
        output_sequence = output_sequence.replace("-", "")

        return output_sequence

    def summarize_current_state(self):
        """
        Logs a summary of current state (delta plddt, delta disorder etc.)
        """
        logger.info(f"The current number of changes is {len(self.mutation_dict)}")
        if len(self.starting_sequence) <= 400:
            logger.info(f"relative delta plddt is {(self.current_plddt - self.starting_plddt) / self.starting_plddt}")
        else:
            logger.info(f"relative delta plddt is NaN")
        logger.info(f"relative delta disorder is "
                    f"{(self.starting_disorder - self.current_disorder) / self.starting_disorder}")
        logger.info(self.current_sequence)
        return

    def score_result(self) -> None:
        """
        Determine if sequence has the lowest disorder probability or best plddt score.
        If so, update scoring variables to keep track of best sequences.

        :return: None
        """
        if len(self.current_sequence) <= 400:
            model = self.fold_sequence(self.current_sequence)
            self.current_plddt = self.get_model_plddt(model)
        else:
            self.current_plddt = 0.0

        self.current_disorder = self.disorder_probability(self.current_sequence, "all")

        if self.current_disorder < self.best_disorder:
            self.best_disorder_sequence = copy.deepcopy(self.current_sequence)
            self.best_disorder_mutation_dict = copy.deepcopy(self.mutation_dict)
            self.best_disorder = copy.deepcopy(self.current_disorder)

        if self.current_plddt > self.best_plddt and len(self.starting_sequence) <= 400:
            self.best_plddt_sequence = copy.deepcopy(self.current_sequence)
            self.best_plddt_mutation_dict = copy.deepcopy(self.mutation_dict)
            self.best_plddt = copy.deepcopy(self.current_plddt)

        return

    def update_output_data(self) -> None:

        self.num_changes_list.append(len(self.mutation_dict))
        self.sequences_list.append(self.current_sequence)
        self.mutation_dicts_list.append(copy.deepcopy(self.mutation_dict))
        self.delta_disorder_list.append(
            float((self.starting_disorder - self.current_disorder) / self.starting_disorder))

        if len(self.starting_sequence) <= 400:
            self.delta_plddt_list.append(float((self.current_plddt - self.starting_plddt) / self.starting_plddt))

        else:
            self.delta_plddt_list.append("NaN")

    def make_output_df(self) -> "dataframe":

        results_df = pd.DataFrame()
        results_df["num_changes"] = self.num_changes_list
        results_df["mutations_dict"] = self.mutation_dicts_list
        results_df["sequence"] = self.sequences_list
        results_df["delta_disorder"] = self.delta_disorder_list
        results_df["delta_plddt"] = self.delta_plddt_list

        exp_details_labels = ["starting_sequence",
                              "max_residues_to_alter",
                              "max_iterations",
                              "protected_residues",
                              "residues_to_modify",
                              "mutations_permitted",
                              "mutations_per_residue",
                              "residues_per_iteration",
                              "evaluation_criterion"]
        exp_details = [self.starting_sequence,
                       self.max_residues_to_alter,
                       self.max_iterations,
                       self.protected_residues,
                       self.residues_to_modify,
                       self.mutation_list,
                       self.gamma,
                       self.epsilon,
                       self.selection_criterion]

        exp_details_df = pd.DataFrame()
        exp_details_df["experimental_details_labels"] = exp_details_labels
        exp_details_df["experimental_details"] = exp_details

        out_df = pd.concat([results_df, exp_details_df], axis=1)

        return out_df

    def basic_disorder_prediction(self):
        """wrapper function for basic disorder prediction"""
        self.load_models()  # load models once
        logger.info("Encoding input sequence.")
        afsm1_pred, afsm2_pred, afsm3_pred, ordinal_list = self.encoder_predictions(self.sequence)
        logger.info("Scoring input sequence for disorder")
        disorder_predictions = self.predict_disorder(self.sequence, afsm1_pred, afsm2_pred, afsm3_pred, ordinal_list)

        return disorder_predictions

    def mutation_tool(self) -> [float, str]:
        """
        wrapper function for the mutation tool part of the class.
        """
        self.load_models()  # load models once
        if len(self.sequence) <= 400:
            model = self.fold_sequence(self.sequence)
            self.starting_plddt = self.get_model_plddt(model)

        else:
            self.starting_plddt = 0.0

        self.starting_disorder = self.disorder_probability(self.sequence, "all")
        logger.info(f"starting sequence: {self.sequence}")
        logger.info(f"starting plddt: {str(self.starting_plddt)}")
        # iterate until max residues are modified
        while len(self.mutation_dict) < self.max_residues_to_alter and self.evolution_counter < self.max_iterations:

            disorder_ranking = self.rank_disorder(self.current_sequence)
            disorder_ranking = self.protect_residues(disorder_ranking)
            logger.info(f"Checking mutations at locations: {disorder_ranking[:self.epsilon]}")
            # generate sequence that maximizes order and plddt from current_sequence
            sequence = self.mutate_residues(disorder_ranking)
            self.evolution_counter += 1
            # end simulation if sequence didn't change or there aren't more sites to mutate
            if sequence == self.current_sequence or len(disorder_ranking) < 1:
                out_df = self.make_output_df()

                return out_df

            if sequence != self.current_sequence:
                self.current_sequence = sequence
                self.score_result()
                self.summarize_current_state()
                self.update_output_data()

        out_df = self.make_output_df()

        return out_df


def parser():
    """A basic argument parser"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-s", "--sequence")
    arg_parser.add_argument("-m", "--max_residues_to_alter")
    arg_parser.add_argument("-i", "--max_iterations")
    arg_parser.add_argument("-p", "--protected_residues")
    arg_parser.add_argument("-o", "--residues_to_modify")
    arg_parser.add_argument("-g", "--gamma")
    arg_parser.add_argument("-e", "--epsilon")
    arg_parser.add_argument("-c", "--selection_criterion")
    return arg_parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()

    PirateEvolutionTool(args.sequence, args.max_residues_to_alter, args.max_iterations,
                        args.protected_residues, args.residues_to_modify,
                        args.gamma, args.epsilon, args.selection_criterion).mutation_tool()
