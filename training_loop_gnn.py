import torch
import torch_geometric
import pickle
import numpy as np

from pyg_triple_update import *
from torch.utils.tensorboard import SummaryWriter
import random

def extract_data(pickle_file="fake_histories_r4_0.pkl"):
    with open(pickle_file, "rb") as f:
        histories = pickle.load(f)
    return histories

def construct_graph(history, device):
    nodes = ConstructNodes(device)
    data = nodes.construct_data(history)
    edges = ConstructEdges(data, device)
    data = edges.generate_edge_indices()
    return data

def preprocess_data(histories, device):
    graphs = []
    for history in histories:
        graphs.append(construct_graph(history, device))
        # print(history['assign_order'])
    return graphs

if __name__ == "__main__":

    # Set device
    device = 'cuda'

    # Set up TensorBoard
    # writer = SummaryWriter()

    # Load data
    histories = extract_data()
    nmr_graphs = preprocess_data(extract_data(), device)

    # Split dataset manually into training and testing
    train_size = int(0.8 * len(nmr_graphs))
    train_data = nmr_graphs[:train_size]
    test_data = nmr_graphs[train_size:]
    train_target = [inp['shift_to_assign'] for inp in histories][:train_size]
    test_target = [inp['shift_to_assign'] for inp in histories][train_size:]

    batch_size = 1
    epochs = 50_000  # Define the number of epochs
    eval_interval = 100  # Evaluate the model every 100 iterations

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create our network and optimizer
    net = TestLayer(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)

    # Interactive plotting
    plt.ion()
    fig, ax = plt.subplots()

    iteration = 0
    for epoch in range(epochs):
        
        for i, xs in enumerate(train_data_loader):
            net.train()
            
            # Break targets into batches manually (if in same order - no shuffling)
            ys = train_target[i*batch_size:(i+1)*batch_size]

            iteration += 1

            _, y_hats = net(xs) 

            # Plots current iteration in batch ##########################################
            ax.cla()
            x_val1, y_val1 = xs['SHIFT'].x[:, 0].tolist(), xs['SHIFT'].x[:, 1].tolist()
            x_val2, y_val2 = xs['RES'].x[:, 3].tolist(), xs['RES'].x[:, 4].tolist()

            ax.scatter(x_val1, y_val1, color='red', label='shift')
            ax.scatter(x_val2, y_val2, color='blue', label='resid')
            
            for j in range(len(x_val1)):
                ax.annotate(j, (x_val1[j], y_val1[j]+0.5), color='red')
                ax.annotate(j, (x_val2[j], y_val2[j]+0.5), color='blue')

            ax.legend()
            ax.set_title(f"Iteration {iteration} - Batch {i}")
            plt.pause(0.01)
            #############################################################################

            loss = 0
            correct = 0
            count = 0
            
            for y_hat, y in zip(y_hats, ys):
                y = torch.tensor([y], dtype=torch.long, device=device)
                loss += torch.nn.functional.cross_entropy(y_hat, y)
                y_pred = torch.argmax(y_hat)

                if y_pred == y:
                    correct += 1
                count += 1
            
            loss = loss / count
            accuracy = correct / count

            print(f"Iteration {iteration} - Loss: {loss.item()} - Accuracy: {accuracy}")

            # writer.add_scalar("Loss/train", loss.item(), iteration)
            # writer.add_scalar("Accuracy/train", accuracy, iteration)

            opt.zero_grad()
            loss.backward()
            opt.step()

    plt.ioff()
    plt.show()
