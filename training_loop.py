import torch
import torcheval
import pickle
import numpy as np
from nmr_transformer import *
from torch.utils.tensorboard import SummaryWriter


def extract_data(pickle_file="fake_histories.pkl"):
    with open(pickle_file, "rb") as f:
        histories = pickle.load(f)
    return histories


def preprocess_data(histories):
    """
    Loops over chosen histories and answers, splits up components, and adds shifts if indices were used.
    """
    predicted_shifts = []
    obs_chemical_shifts = []
    noes = []
    close_dist = []
    assignments = []
    peak_to_assign = []
    correct_answer = []

    # batch size needs to be reworked in here (not randomized - selects from start up until 'batch_size' index)
    for history in histories:
        # Predicted shifts
        predicted_shifts_temp = [(i.H1, i.N15) for i in history["coords"]]
        predicted_shifts.append(predicted_shifts_temp)
        # Observed shifts
        obs_chemical_shifts_temp = [(i.H1, i.N15) for i in history["actual_shifts"]]
        obs_chemical_shifts.append(obs_chemical_shifts_temp)
        # NOEs
        noes.append(history["noes"])

        # Close distances (duplicated for reverse pair order)
        close = []
        for i in history["connectivity"]:
            atom1 = predicted_shifts_temp[i.atom1]
            atom2 = predicted_shifts_temp[i.atom2]
            close.append((atom1[0], atom1[1], atom2[0], atom2[1]))
            close.append((atom2[0], atom2[1], atom1[0], atom1[1]))
        close_dist.append(close)

        # Assignments
        assign = []
        for i, j in list(history["assignments"].items()):
            shift1 = obs_chemical_shifts_temp[i]
            shift2 = predicted_shifts_temp[j]
            assign.append((shift1[0], shift1[1], shift2[0], shift2[1]))
        if assign:
            assignments.append(assign)
        else:
            assignments.append(None)

        # Peak to assign
        peak_to_assign.append(history["assign_shift"])
        correct_answer.append(history["assign_shift"])

    return (
        predicted_shifts,
        obs_chemical_shifts,
        noes,
        close_dist,
        assignments,
        peak_to_assign,
        correct_answer,
    )


def organize_nmr_inputs(histories):
    """
    Sets up NMR inputs as list of named tuples.
    """
    (
        predicted_shifts,
        actual_shifts,
        noes,
        close_dist,
        assignments,
        assign_peak,
        correct_answer,
    ) = preprocess_data(histories)

    nmr_inputs = [
        (
            NMRInput(
                obs_chemical_shifts=torch.tensor(actual_shifts[i], dtype=torch.float),
                pred_chemical_shifts=torch.tensor(predicted_shifts[i], dtype=torch.float),
                obs_noes=torch.tensor(noes[i], dtype=torch.float),
                close_distances=torch.tensor(close_dist[i], dtype=torch.float),
                assigned_peaks=torch.tensor(assignments[i], dtype=torch.float) if assignments[i] else None,
                peak_to_assign=assign_peak[i],
            ),
            correct_answer[i],
        )
        for i in range(len(histories))
    ]
    return nmr_inputs


if __name__ == "__main__":
    # set up tensor board
    writer = SummaryWriter()

    # load data
    nmr_inputs = organize_nmr_inputs(extract_data())

    # create our network and optimizer
    net = NMRTransformer(dropout=0.0)
    net.compile()
    opt = torch.optim.Adam(net.parameters())
    policy_loss = torch.nn.CrossEntropyLoss()

    # training loop
    iteration = 0
    while True:
        iteration += 1
        xs = [inp[0] for inp in nmr_inputs]
        ys = [inp[1] for inp in nmr_inputs]

        y_hats, _ = net(xs)

        loss = 0
        count = 0
        for y_hat, y in zip(y_hats, ys):
            delta = policy_loss(y_hat, torch.tensor(y))
            loss += delta
            count += 1
        
        loss = loss / count

        correct = 0
        trials = 0
        for y_hat, y in zip(y_hats, ys):
            y_pred = torch.argmax(y_hat)
            if y_pred == y:
                correct += 1
            trials += 1
        accuracy = correct / trials

        print(loss.item(), accuracy)

        writer.add_scalar("Loss/train", loss.item(), iteration)
        writer.add_scalar("Loss/accuracy", accuracy, iteration)
        opt.zero_grad()
        loss.backward()
        opt.step()

