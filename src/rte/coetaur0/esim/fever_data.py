"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2018.

import string
import torch
import numpy as np
import json

from collections import Counter
from torch.utils.data import Dataset


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None,
                 concat_premises=True):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos
        self.concat_premises = concat_premises

    def preprocess_premises_concat(self, premise, parentheses_table, punct_table):
        premise = " ".join(premise)

        # Remove '(' and ')' from the premises
        premise = premise.translate(parentheses_table)

        if self.lowercase:
            premise = premise.lower()

        if self.ignore_punctuation:
            premise = premise.translate(punct_table)

        # Each premise is split into a list of words.
        return [w for w in premise.rstrip().split()
                                 if w not in self.stopwords]


    def preprocess_premises_flat(self, premise, parentheses_table, punct_table):
        """
        preprocess the premises (evidence) without concatenating all the 
        sentences into one sentence
        """
        # Remove '(' and ')' from the premises
        premise = [p.translate(parentheses_table) for p in premise]

        if self.lowercase:
            premise = [p.lower() for p in premise]

        if self.ignore_punctuation:
            premise = [p.translate(punct_table) for p in premise]

        # Each premise is split into a list of words.
        return [[w for w in p.rstrip().split()
                                 if w not in self.stopwords] for p in premise]


    def read_data(self, filepath, testing=False):
        """
        Read the premises, hypotheses and labels from some NLI dataset's
        file and return them in a dictionary. The file should be in the same
        form as FEVER's .jsonl files.

        Args:
            filepath: The path to a file containing some premises, hypotheses
                and labels that must be read. The file should be formatted in
                the same way as the FEVER dataset.

        Returns:
            A dictionary containing three lists, one for the premises, one for
            the hypotheses, and one for the labels in the input data.
        """
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, labels = [], [], [], []

            # Translation tables to remove parentheses and punctuation from
            # strings.
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

            # Ignore the headers on the first line of the file.
            #next(input_data)

            for line in input_data:
                line = json.loads(line)

                pair_id = line["id"]
                premise = line["evidence"]
                hypothesis = line["claim"]

                max_prem = premise[0]
                if not testing:
                    for i in range(1, len(premise)):
                        if len(premise[i]) > len(max_prem):
                            max_prem = premise[i]
                    premise = [p[3] for p in max_prem]
                    labels.append(line["label"])

                if self.concat_premises:
                    premise = self.preprocess_premises_concat(premise, parentheses_table, punct_table)
                else:
                    premise = self.preprocess_premises_flat(premise, parentheses_table, punct_table)

                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append(premise)
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                ids.append(pair_id)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": ["NOT ENOUGH INFO"] * len(premises) if testing else labels}

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        if self.concat_premises:
            [words.extend(sentence) for sentence in data["premises"]]
        else:
            for sample in data["premises"]:
                for sentence in sample:
                    for word in sentence:
                        words.append(word)
        [words.extend(sentence) for sentence in data["hypotheses"]]

        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}
        print("labels dict: ", self.labeldict)

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "labels": []}

        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["ids"].append(data["ids"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            if self.concat_premises:
                indices = self.words_to_indices(premise)
            else:
                indices = [self.words_to_indices(p) for p in premise]
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None,
                 premises_concat=True):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.premises_concat = premises_concat
        self.max_premise_length = max_premise_length
        if self.premises_concat:
            self.premises_lengths = [len(seq) for seq in data["premises"]]
        else: 
            self.premises_lengths = [[len(seq) for seq in sen] for sen in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            if self.premises_concat:
                self.max_premise_length = max(self.premises_lengths)
            else:
                self.max_premise_length = max([max(l) for l in self.premises_lengths])

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        if self.premises_concat:
            premises = torch.ones((self.num_sequences, self.max_premise_length),
                dtype=torch.long) * padding_idx
        else:
            premises = [None] * self.num_sequences

        self.data = {"ids": [],
                     "premises": premises,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):
            self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            if self.premises_concat:
                self.data["premises"][i][:end] = torch.tensor(premise[:end])
            else:
                self.data["premises"][i] = [None] * len(data["premises"][i])
                for j, sentence in enumerate(premise, 0):
                    self.data["premises"][i][j] = torch.ones(self.max_premise_length, dtype=torch.long) * padding_idx
                    sen_end = min(len(sentence), self.max_premise_length)
                    self.data["premises"][i][j][:sen_end] = torch.tensor(sentence[:sen_end], dtype=torch.long)

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        prem_lens = [min(sen_len, self.max_premise_length)
                                   for sen_len in self.premises_lengths[index]] \
                                           if not self.premises_concat \
                                           else min(self.premises_lengths[index], self.max_premise_length)
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": prem_lens,
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index]}
