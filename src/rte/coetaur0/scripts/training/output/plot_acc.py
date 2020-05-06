#!/usr/bin/env python

import matplotlib.pyplot as plt
import argparse
import os

train_file = "./model_agg/agg_two_hidden/train_acc"
dev_file = "./model_agg/agg_two_hidden/val_acc"
title = "Accuracy Graph"
ylabel = "Accuracy (%)"
save_path = "./model_agg/agg_two_hidden/train_dev_loss.png"

##### MAIN

class Figure:

    def __init__(self, x_vals, y_vals, xlabel, ylabel, title=None, legend=None, color=None):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.legend = legend
        self.color = color


def add_vals_to_plot(x_vals, y_vals, color):
    if color:
        plt.plot(x_vals, y_vals, color)
    else:
        plt.plot(x_vals, y_vals)


def plot_single(figure: "Figure", save_path):
    fig = plt.figure()
    add_vals_to_plot(figure.x_vals, figure.y_vals, figure.color)
    plt.ylabel(figure.ylabel)
    plt.xlabel(figure.xlabel)
    plt.legend([figure.legend])
    plt.show()
    fig.savefig(save_path)


def plot_multiple(figures: list, save_path):
    legends = []
    fig = plt.figure()
    for figure in figures:
        add_vals_to_plot(figure.x_vals, figure.y_vals, figure.color)
        legends.append(figure.legend)
    plt.ylabel(figures[0].ylabel)
    plt.xlabel(figures[0].xlabel)
    plt.legend(legends)
    plt.show()
    fig.savefig(save_path)


def read_vals(path):
    vals = []
    with open(path, 'r') as vals_file:
        vals.extend([float(line) for line in vals_file.readlines()])
    return vals


if __name__ == "__main__":
    train_vals = [v * 100 for v in read_vals(train_file)]
    dev_vals = [v * 100 for v in read_vals(dev_file)]
    epochs = range(1, len(train_vals) + 1)
    epoch_lbl = "Epoch"
    train_lbl = "Training"
    dev_lbl = "Development"

    train_fig = Figure(x_vals=epochs, y_vals=train_vals,
            xlabel=epoch_lbl, ylabel=ylabel,
            title=title, legend=train_lbl, color="g")
    dev_fig = Figure(x_vals=epochs, y_vals=dev_vals,
            xlabel=epoch_lbl, ylabel=ylabel,
            title=title, legend=dev_lbl, color="r")

    plot_multiple([train_fig, dev_fig], save_path)

    
