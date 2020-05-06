"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import time
import pickle
import argparse
import torch
import os
import json

from torch.utils.data import DataLoader
from rte.coetaur0.esim.fever_data import NLIDataset
from rte.coetaur0.esim.model import ESIM
from rte.coetaur0.esim.utils import correct_predictions
from rte.coetaur0.scripts.training.utils import forward_with_aggregator
from rte.tmp.lbl_agg import LabelAggregator
from rte.coetaur0.scripts.training.train_fever_agg import custom_collate


def append_preds_to_file(preds, out_file):
    for output in preds:
        p = output.item()
        label_text = "NOT ENOUGH INFO"
        if p == 0:
            label_text = "SUPPORTS"
        elif p == 1:
            label_text = "REFUTES"
        with open(out_file, "a+") as pred_file:
            pred_file.write(label_text + "\n")


def test(model, dataloader, out_file, num_sentences=5, use_aggregator=False,
        aggregator=None, premises_concat=True):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            if premises_concat:
                premises = batch["premise"].to(device)
                premises_lengths = batch["premise_length"].to(device)
                hypotheses = batch["hypothesis"].to(device)
                hypotheses_lengths = batch["hypothesis_length"].to(device)
                labels = batch["label"].to(device)
            else:
                premises = batch["premise"]
                premises_lengths = batch["premise_length"]
                hypotheses = batch["hypothesis"]
                hypotheses_lengths = batch["hypothesis_length"]
                labels = batch["label"].to(device)


            if use_aggregator:
                aggregator.eval()
                probs, _, _, _ = \
                    forward_with_aggregator(model, aggregator, premises,
                                            premises_lengths, hypotheses,
                                            hypotheses_lengths, labels,
                                            num_sentences, device)
            else:
                _, probs = model(premises,
                                 premises_lengths,
                                 hypotheses,
                                 hypotheses_lengths)

            _, out_classes = probs.max(dim=1)

            append_preds_to_file(out_classes, out_file)


def main(test_file, pretrained_file, batch_size, out_file,
         num_sentences=5, use_aggregator=False, aggregator_file=None,
         premises_concat=True, max_premise_length=100):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file)

    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
    hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
    num_classes = checkpoint["model"]["_classification.4.weight"].size(0)

    print("\t* Loading test data...")
    with open(test_file, "rb") as pkl:
        test_data = NLIDataset(pickle.load(pkl), max_premise_length=max_premise_length,
                premises_concat=premises_concat)

    if premises_concat:
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    else:
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, collate_fn=custom_collate)


    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint["model"])

    aggregator = None
    if use_aggregator:
        agg_checkpoint = torch.load(aggregator_file)
        aggregator = LabelAggregator(num_labels=num_classes,
                                     num_sentences=num_sentences).to(device)
        aggregator.load_state_dict(agg_checkpoint["model"])

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    test(model, test_loader, out_file, num_sentences, use_aggregator, aggregator, premises_concat)


if __name__ == "__main__":

    config_path = "../../config/testing/fever_testing.json"
    default_sen_config = "../../config/sentence_params.json"
    script_dir = os.path.dirname(os.path.realpath(__file__))

    config_path = os.path.join(script_dir, config_path)
    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    sen_config_path = os.path.join(script_dir, default_sen_config)
    with open(os.path.normpath(sen_config_path), 'r') as sen_config_file:
        sen_config = json.load(sen_config_file)

    esim_ckp = os.path.join(script_dir, config["esim_checkpoint"])
    agg_ckp = os.path.join(script_dir, config["aggregator_checkpoint"])

    print("Sentence config is: ", sen_config)

    main(config["test_data"],
         esim_ckp,
         config["batch_size"],
         config["out_file"],
         sen_config["num_sentences"],
         config["use_aggregator"],
         agg_ckp,
         sen_config["premises_concat"],
         sen_config["max_premise_length"])
