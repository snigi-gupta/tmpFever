"""
Train the ESIM model on the preprocessed SNLI dataset.
"""
# Aurelien Coet, 2018.

import os
import argparse
import pickle
import torch
import json

import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader
from rte.coetaur0.esim.fever_data import NLIDataset
from rte.coetaur0.esim.model import ESIM
from rte.tmp.lbl_agg import LabelAggregator

from .utils import validate_with_agg, train_with_agg


def custom_collate(batch):
    res = {"premise": [], "premise_length": [], "hypothesis": [],
           "hypothesis_length": [],
           "label": torch.zeros(len(batch), dtype=torch.long)}
    for i, sample in enumerate(batch, 0):
        res["premise"].append(sample["premise"])
        res["premise_length"].append(sample["premise_length"])
        res["hypothesis"].append(sample["hypothesis"])
        res["hypothesis_length"].append(sample["hypothesis_length"])
        res["label"][i] = sample["label"]
    return res


def main(train_file,
         valid_file,
         embeddings_file,
         target_dir,
         hidden_size=300,
         dropout=0.5,
         num_classes=3,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None,
         max_premises_length=None,
         premises_concat=True,
         agg_checkpoint=None,
         num_sentences=None):
    """
    Train the ESIM model on the SNLI dataset.

    Args:
        train_file: A path to some preprocessed data that must be used
            to train the model.
        valid_file: A path to some preprocessed data that must be used
            to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        target_dir: The path to a directory where the trained model must
            be saved.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        num_classes: The number of classes in the output of the model.
            Defaults to 3.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        lr: The learning rate for the optimizer. Defaults to 0.0004.
        patience: The patience to use for early stopping. Defaults to 5.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
        max_premises_length: the maximum number of words to use for each sentence
        premises_concat: indicates whether the sentences have been
            concatenated together or kept separate.
        agg_checkpoint: the checkpoint from which the aggregator should resume training
        num_sentences: the maximum number of sentences included in each sample
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    with open(train_file, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl), max_premise_length=max_premises_length,
                                premises_concat=premises_concat)

    train_loader = DataLoader(train_data, shuffle=True, collate_fn=custom_collate, batch_size=batch_size)

    print("\t* Loading validation data...")
    with open(valid_file, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl), max_premise_length=max_premises_length,
                                premises_concat=premises_concat)

    valid_loader = DataLoader(valid_data, shuffle=False, collate_fn=custom_collate, batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float) \
            .to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    aggregator = LabelAggregator(num_labels=num_classes,
                                 num_sentences=num_sentences).to(device)
    # -------------------- Preparation for training  ------------------- #
    agg_criterion = nn.CrossEntropyLoss()
    agg_params = aggregator.parameters()
    print("type of parameters: ", type(agg_params))
    agg_optimizer = torch.optim.Adam(agg_params, lr=lr)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #epochs_count = checkpoint["epochs_count"]
        #train_losses = checkpoint["train_losses"]
        #valid_losses = checkpoint["valid_losses"]

    if agg_checkpoint:
        agg_checkpoint = torch.load(agg_checkpoint)
        start_epoch = agg_checkpoint["epoch"] + 1
        best_score = agg_checkpoint["best_score"]

        print("\t* Training will continue on existing aggregator model from epoch {}..."
              .format(start_epoch))

        aggregator.load_state_dict(agg_checkpoint["model"])
        #agg_optimizer.load_state_dict(agg_checkpoint["optimizer"])
        #epochs_count = agg_checkpoint["epochs_count"]
        #train_losses = agg_checkpoint["train_losses"]
        #valid_losses = agg_checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate_with_agg(model,
                                                      valid_loader,
                                                      criterion,
                                                      aggregator,
                                                      agg_criterion,
                                                      num_sentences)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy * 100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train_with_agg(model,
                                                                train_loader,
                                                                optimizer,
                                                                criterion,
                                                                epoch,
                                                                max_grad_norm,
                                                                aggregator,
                                                                agg_optimizer,
                                                                agg_criterion,
                                                                num_sentences)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate_with_agg(model,
                                                                   valid_loader,
                                                                   criterion,
                                                                   aggregator,
                                                                   agg_criterion,
                                                                   num_sentences)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))
            torch.save({"epoch": epoch,
                        "model": aggregator.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best_agg.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))
        torch.save({"epoch": epoch,
                    "model": aggregator.state_dict(),
                    "best_score": best_score,
                    "optimizer": agg_optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "agg_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    plt.show()


if __name__ == "__main__":
    default_config = "../../config/training/fever_training.json"
    default_sen_config = "../../config/sentence_params.json"
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="Train the ESIM model on SNLI")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument(
        "--sentence_config",
        default=default_sen_config,
        help="Path to a configuration file for sentence params"
    )
    parser.add_argument("--checkpoint",
                        default=os.path.join(script_dir, "../../data/checkpoints/fever/best.pth.tar"),
                        help="Path to a checkpoint file to resume training")
    parser.add_argument("--agg_checkpoint",
                        default=os.path.join(script_dir, "../../data/checkpoints/fever/best_agg.pth.tar"),
                        help="Path to a checkpoint file to resume training for label aggregator")

    args = parser.parse_args()


    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    sen_config_path = args.sentence_config
    if args.sentence_config == default_sen_config:
        sen_config_path = os.path.join(script_dir, args.sentence_config)

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)
    with open(os.path.normpath(sen_config_path), "r") as sen_cfg_file:
        sen_config = json.load(sen_cfg_file)

    print("Sentence config is: ", sen_config)

    main(os.path.normpath(os.path.join(script_dir, config["train_data"])),
         os.path.normpath(os.path.join(script_dir, config["valid_data"])),
         os.path.normpath(os.path.join(script_dir, config["embeddings"])),
         os.path.normpath(os.path.join(script_dir, config["target_dir"])),
         config["hidden_size"],
         config["dropout"],
         config["num_classes"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"],
         args.checkpoint,
         sen_config["max_premise_length"],
         sen_config["premises_concat"],
         args.agg_checkpoint,
         sen_config["num_sentences"])
