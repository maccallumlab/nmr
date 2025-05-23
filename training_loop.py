import torch
import pickle
import numpy as np
from nmr_transformer import *
from torch.utils.tensorboard import SummaryWriter
import random



def extract_data(pickle_file="fake_histories_r20_1.pkl"):
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

    for history in histories:
        # Predicted shifts
        predicted_shifts_temp = [(i.H1, i.N15) for i in history["coordinates"]]
        predicted_shifts.append(predicted_shifts_temp)
        # Observed shifts
        obs_chemical_shifts_temp = [(i.H1, i.N15) for i in history["obs_chemical_shifts"]]
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
        peak_to_assign.append(history["shift_to_assign"])
        correct_answer.append(history["shift_to_assign"])

    return (
        predicted_shifts,
        obs_chemical_shifts,
        noes,
        close_dist,
        assignments,
        peak_to_assign,
        correct_answer
    )


def organize_nmr_inputs(histories, device):
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
                obs_chemical_shifts=torch.tensor(actual_shifts[i], dtype=torch.float, device=device),
                pred_chemical_shifts=torch.tensor(predicted_shifts[i], dtype=torch.float, device=device),
                obs_noes=torch.tensor(noes[i], dtype=torch.float, device=device),
                close_distances=torch.tensor(close_dist[i], dtype=torch.float, device=device),
                assigned_peaks=torch.tensor(assignments[i], dtype=torch.float, device=device) if assignments[i] else None,
                peak_to_assign=assign_peak[i],
            ),
            correct_answer[i],
        )
        for i in range(len(histories))
    ]
    return nmr_inputs



if __name__ == "__main__":
    
    # Set device
    # device = "mps"
    device ='cuda'

    # Set up TensorBoard
    writer = SummaryWriter()

    # Load data
    nmr_inputs = organize_nmr_inputs(extract_data(), device=device)

    # Shuffle the data
    random.shuffle(nmr_inputs)

    # Split dataset manually into training and testing
    train_size = int(0.8 * len(nmr_inputs))
    train_data = nmr_inputs[:train_size]
    test_data = nmr_inputs[train_size:]

    # Create our network and optimizer
    net = NMRTransformer(dropout=0.1, n_layers=4, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    policy_loss = torch.nn.CrossEntropyLoss()

    # Training loop
    batch_size = 10
    epochs = 500_000  # Define the number of epochs
    eval_interval = 100  # Evaluate the model every 100 iterations

    iteration = 0
    for epoch in range(epochs):
        random.shuffle(train_data)  # Shuffle training data each epoch

        # Break training data into batches manually
        for i in range(0, len(train_data), batch_size):
            net.train()
            batch = train_data[i:i + batch_size]
            xs = [inp[0] for inp in batch]
            ys = [inp[1] for inp in batch]

            iteration += 1

            y_hats, _ = net(xs)

            loss = 0
            correct = 0
            count = 0

            for y_hat, y in zip(y_hats, ys):
                y = torch.tensor(y, device=device)
                loss += policy_loss(y_hat, y)

                y_pred = torch.argmax(y_hat)
                if y_pred == y:
                    correct += 1
                count += 1

            loss = loss / count
            accuracy = correct / count

            print(f"Iteration {iteration} - Loss: {loss.item()} - Accuracy: {accuracy}")

            writer.add_scalar("Loss/train", loss.item(), iteration)
            writer.add_scalar("Accuracy/train", accuracy, iteration)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Evaluate on test set periodically
            if iteration % eval_interval == 0:
                net.eval()
                test_loss = 0
                test_correct = 0
                test_trials = 0

                with torch.no_grad():
                    for j in range(0, len(test_data), batch_size):
                        test_batch = test_data[j:j + batch_size]
                        xs = [inp[0] for inp in test_batch]
                        ys = [inp[1] for inp in test_batch]

                        test_y_hats, _ = net(xs)
                        
                        batch_loss = 0
                        for y_hat, y in zip(test_y_hats, ys):
                            y = torch.tensor(y, device=device)
                            batch_loss += policy_loss(y_hat, y).item()
                            if torch.argmax(y_hat) == y:
                                test_correct += 1
                            test_trials += 1
                        
                        test_loss += batch_loss / len(test_batch)
                        
                test_loss /= (len(test_data) / batch_size)
                test_accuracy = test_correct / test_trials

                print(f"Test Loss: {test_loss} - Test Accuracy: {test_accuracy}")
                writer.add_scalar("Loss/test", test_loss, iteration)
                writer.add_scalar("Accuracy/test", test_accuracy, iteration)
