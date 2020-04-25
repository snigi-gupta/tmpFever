"""
Utility functions for training and validating models.
"""

import time
import torch

import torch.nn as nn

from tqdm import tqdm
from rte.coetaur0.esim.utils import correct_predictions


def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, probs = model(premises,
                              premises_lengths,
                              hypotheses,
                              hypotheses_lengths)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy


def validate_with_agg(model, dataloader, criterion, aggregator, agg_criterion):
    num_sentences = 5
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    total_num_sens = 0
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"]
            premises_lengths = batch["premise_length"]
            hypotheses = batch["hypothesis"]
            hypotheses_lengths = batch["hypothesis_length"]
            labels = batch["label"].to(device)

            # flat_batch_prems = torch.zeros(0, dtype=torch.long).to(device)
            # flat_batch_prem_lens = torch.zeros(0, dtype=torch.long).to(device)
            # flat_batch_hypos = torch.zeros(0, dtype=torch.long).to(device)
            # flat_batch_hypo_lens = torch.zeros(0, dtype=torch.long).to(device)
            # flat_batch_labels = torch.zeros(0, dtype=torch.long).to(device)
            #
            # for i in range(len(premises)):
            #     prem = premises[i][:num_sentences]
            #     hypo = hypotheses[i].to(device)
            #     prem_len = premises_lengths[i][:num_sentences]
            #     hypo_len = torch.tensor(hypotheses_lengths[i]).to(device)
            #     label = labels[i].to(device)
            #
            #     # feed each of the sentences in prem to the model separately
            #     sen_scores = torch.zeros(0)
            #     hypo_len = torch.tensor(hypo_len, dtype=torch.long).unsqueeze(0).unsqueeze(1)
            #     label = torch.tensor(label, dtype=torch.long).unsqueeze(0).unsqueeze(1)
            #     hypo = hypo.unsqueeze(0)
            #     for sentence, sen_len in zip(prem, prem_len):
            #         sen_len = torch.tensor(sen_len, dtype=torch.long).unsqueeze(0).unsqueeze(1)
            #         sentence = sentence.unsqueeze(0)
            #         flat_batch_prems = torch.cat([flat_batch_prems, sentence.to(device)])
            #         flat_batch_prem_lens = torch.cat([flat_batch_prem_lens, sen_len.to(device)])
            #         flat_batch_hypos = torch.cat([flat_batch_hypos, hypo.to(device)])
            #         flat_batch_hypo_lens = torch.cat([flat_batch_hypo_lens, hypo_len.to(device)])
            #         flat_batch_labels = torch.cat([flat_batch_labels, label.to(device)])
            #
            # logits, probs = model(flat_batch_prems.to(device),
            #                       flat_batch_prem_lens.squeeze().to(device),
            #                       flat_batch_hypos.to(device),
            #                       flat_batch_hypo_lens.squeeze().to(device))
            # total_num_sens += logits.shape[0]
            # num_classes = logits.shape[1]
            # input_dim = num_sentences * num_classes * 2
            # rolled_logits = torch.zeros((len(labels), input_dim))
            # sen_end_index = 0
            # for i in range(len(premises)):
            #     prem_len = premises_lengths[i][:num_sentences]
            #     prem_len_len = len(prem_len)
            #     prems_logits = logits[sen_end_index: sen_end_index + prem_len_len].flatten()
            #     prems_probs = probs[sen_end_index: sen_end_index + prem_len_len].flatten()
            #     rolled_logits[i, :prems_logits.shape[0]] = prems_logits
            #     rolled_logits[i, input_dim // 2: (input_dim // 2) + prems_logits.shape[0]] = prems_probs
            #     sen_end_index += prem_len_len
            # agg_logits, agg_probs = aggregator(rolled_logits.to(device))
            #
            # labels = labels.to(device)
            # loss = criterion(logits, flat_batch_labels.squeeze().to(device))
            # agg_loss = agg_criterion(agg_logits, labels)

            _, agg_probs, loss, _ = \
                forward_on_aggregator_with_loss(model, aggregator,
                                                criterion, agg_criterion,
                                                premises, premises_lengths,
                                                hypotheses, hypotheses_lengths,
                                                labels, num_sentences, device)

            running_loss += loss.item()
            running_accuracy += correct_predictions(agg_probs.to(device), labels)

    epoch_time = time.time() - epoch_start
    # epoch_loss = running_loss / len(dataloader)
    epoch_loss = running_loss / total_num_sens
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    with open("val_acc", "a+") as f:
        f.write(str(epoch_accuracy) + '\n')

    return epoch_time, epoch_loss, epoch_accuracy


def train_with_agg(model,
                   dataloader,
                   optimizer,
                   criterion,
                   epoch_number,
                   max_gradient_norm,
                   aggregator,
                   agg_optimizer,
                   agg_criterion):
    num_sentences = 5
    model.train()
    aggregator.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    total_num_sens = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        agg_optimizer.zero_grad()

        agg_loss, agg_probs, loss, sens_in_batch = \
            forward_on_aggregator_with_loss(model, aggregator,
                                            criterion, agg_criterion,
                                            premises, premises_lengths,
                                            hypotheses, hypotheses_lengths,
                                            labels, num_sentences, device)

        total_num_sens += sens_in_batch

        agg_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(aggregator.parameters(), max_gradient_norm)
        agg_optimizer.step()

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(agg_probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    # epoch_loss = running_loss / len(dataloader)
    epoch_loss = running_loss / total_num_sens
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    with open("train_acc", "a+") as f:
        f.write(str(epoch_accuracy) + '\n')

    return epoch_time, epoch_loss, epoch_accuracy


def forward_on_aggregator_with_loss(model, aggregator, criterion,
                                    agg_criterion, premises,
                                    premises_lengths, hypotheses,
                                    hypotheses_lengths, labels,
                                    num_sentences, device):
    agg_probs, agg_logits, logits, flat_batch_labels = forward_with_aggregator(model, aggregator, premises,
                                                                               premises_lengths, hypotheses,
                                                                               hypotheses_lengths, labels,
                                                                               num_sentences, device)
    loss = criterion(logits, flat_batch_labels.squeeze().to(device))
    agg_loss = agg_criterion(agg_logits, labels)
    return agg_loss, agg_probs, loss, logits.shape[0]


def forward_with_aggregator(model, aggregator, premises,
                            premises_lengths, hypotheses,
                            hypotheses_lengths, labels,
                            num_sentences, device):
    flat_batch_prems = torch.zeros(0, dtype=torch.long).to(device)
    flat_batch_prem_lens = torch.zeros(0, dtype=torch.long).to(device)
    flat_batch_hypos = torch.zeros(0, dtype=torch.long).to(device)
    flat_batch_hypo_lens = torch.zeros(0, dtype=torch.long).to(device)
    flat_batch_labels = torch.zeros(0, dtype=torch.long).to(device)
    for i in range(len(premises)):
        prem = premises[i][:num_sentences]
        hypo = hypotheses[i].to(device)
        prem_len = premises_lengths[i][:num_sentences]
        hypo_len = torch.tensor(hypotheses_lengths[i]).to(device)
        label = labels[i].to(device)

        # feed each of the sentences in prem to the model separately
        hypo_len = torch.tensor(hypo_len, dtype=torch.long).unsqueeze(0).unsqueeze(1)
        label = torch.tensor(label, dtype=torch.long).unsqueeze(0).unsqueeze(1)
        hypo = hypo.unsqueeze(0)
        for sentence, sen_len in zip(prem, prem_len):
            sen_len = torch.tensor(sen_len, dtype=torch.long).unsqueeze(0).unsqueeze(1)
            sentence = sentence.unsqueeze(0)
            flat_batch_prems = torch.cat([flat_batch_prems, sentence.to(device)])
            flat_batch_prem_lens = torch.cat([flat_batch_prem_lens, sen_len.to(device)])
            flat_batch_hypos = torch.cat([flat_batch_hypos, hypo.to(device)])
            flat_batch_hypo_lens = torch.cat([flat_batch_hypo_lens, hypo_len.to(device)])
            flat_batch_labels = torch.cat([flat_batch_labels, label.to(device)])
    logits, probs = model(flat_batch_prems.to(device),
                          flat_batch_prem_lens.squeeze().to(device),
                          flat_batch_hypos.to(device),
                          flat_batch_hypo_lens.squeeze().to(device))
    num_classes = logits.shape[1]
    input_dim = num_sentences * num_classes * 2
    rolled_logits = torch.zeros((len(labels), input_dim))
    sen_end_index = 0
    for i in range(len(premises)):
        prem_len = premises_lengths[i][:num_sentences]
        prem_len_len = len(prem_len)
        prems_logits = logits[sen_end_index: sen_end_index + prem_len_len].flatten()
        prems_probs = probs[sen_end_index: sen_end_index + prem_len_len].flatten()
        rolled_logits[i, :prems_logits.shape[0]] = prems_logits
        rolled_logits[i, input_dim // 2: (input_dim // 2) + prems_logits.shape[0]] = prems_probs
        sen_end_index += prem_len_len
    agg_logits, agg_probs = aggregator(rolled_logits.to(device))
    return agg_probs, agg_logits, logits, flat_batch_labels
