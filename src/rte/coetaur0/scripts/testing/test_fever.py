"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import time
import pickle
import argparse
import torch

from torch.utils.data import DataLoader
from rte.coetaur0.esim.fever_data import NLIDataset
from rte.coetaur0.esim.model import ESIM
from rte.coetaur0.esim.utils import correct_predictions


test_data = "/home/kikuchio/nlp/src/data/tmp/preprocessed/test_data.pkl"
checkpoint = "/home/kikuchio/nlp/src/rte/coetaur0/data/checkpoints/fever/best.pth.tar"
batch_size = 32
out_file = "/home/kikuchio/nlp/src/rte/coetaur0/data/test_preds.txt"


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


def test(model, dataloader, out_file):
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
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            _, out_classes = probs.max(dim=1)

            append_preds_to_file(out_classes, out_file)


def main(test_file, pretrained_file, out_file, batch_size=32):
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
        test_data = NLIDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint["model"])

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    test(model, test_loader, out_file)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Test the ESIM model on\
    # some dataset")
    #parser.add_argument("test_data",
    #                    help="Path to a file containing preprocessed test data", type=str,
    #                    default="../../../../../data/tmp/preprocessed/test_data.pkl")
    #parser.add_argument("checkpoint", type=str,
    #                    help="Path to a checkpoint with a pretrained model",
    #                    default="../../data/checkpoints/best.pth.tar")
    #parser.add_argument("--batch_size", type=int, default=32,
    #                    help="Batch size to use during testing")
    #args = parser.parse_args()


    main(test_data,
         checkpoint, 
         out_file,
         batch_size)
