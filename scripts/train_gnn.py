import torch
from pprint import pp
import sys
from pathlib import Path
import argparse

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.construct import construct_graph
from nmr.models import NMRNet
from nmr.nmr_gym.io import load_histories
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt


def extract_data(pickle_file):
    """Load histories and flatten into list of (state_dict, action, reward) tuples."""
    histories = load_histories(pickle_file)

    # Flatten trajectories into individual training examples
    # Each trajectory contains steps: (state_dict, action, reward)
    examples = []
    for trajectory in histories.trajectories:
        for state_dict, action, value in trajectory:
            examples.append((state_dict, action, value))

    return examples


def preprocess_data(examples, device):
    """Convert state dictionaries to graphs and return actions and values separately."""
    graphs = []
    actions = []
    values = []

    for state_dict, action, value in examples:
        graph = construct_graph(state_dict, device)
        graphs.append(graph)
        actions.append(action)
        values.append(value)

    return graphs, actions, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN model on NMR assignment histories")
    parser.add_argument(
        "--histories",
        type=str,
        required=True,
        help="Path to the pickle file containing training histories"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training (default: cpu)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50000,
        help="Number of training epochs (default: 50000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate model every N iterations (default: 100)"
    )

    args = parser.parse_args()

    # Set device
    device = args.device

    # Load data
    examples = extract_data(args.histories)
    nmr_graphs, actions, rewards = preprocess_data(examples, device)

    batch_size = args.batch_size
    epochs = args.epochs
    eval_interval = args.eval_interval

    data_loader = DataLoader(nmr_graphs, batch_size=batch_size, shuffle=False)

    # Create our network and optimizer
    net = NMRNet(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Interactive plotting
    plt.ion()
    fig, ax = plt.subplots()

    iteration = 0
    for epoch in range(epochs):

        for i, xs in enumerate(data_loader):
            net.train()
            iteration += 1

            # Break targets into batches manually (if in same order - no shuffling)
            ys = actions[i * batch_size : (i + 1) * batch_size]

            _, _, policies = net(xs)

            # Plots current iteration in batch ##########################################
            ax.cla()
            x_val1, y_val1 = xs["SHIFT"].x[:, 0].tolist(), xs["SHIFT"].x[:, 1].tolist()
            x_val2, y_val2 = xs["RES"].x[:, 3].tolist(), xs["RES"].x[:, 4].tolist()

            ax.scatter(x_val1, y_val1, color="red", label="shift")
            ax.scatter(x_val2, y_val2, color="blue", label="resid")

            for j in range(len(x_val1)):
                ax.annotate(j, (x_val1[j], y_val1[j] + 0.5), color="red")
                ax.annotate(j, (x_val2[j], y_val2[j] + 0.5), color="blue")

            ax.legend()
            ax.set_title(f"Iteration {iteration} - Batch {i}")
            plt.pause(0.01)
            #############################################################################

            loss = 0
            correct = 0
            count = 0

            for policy, y in zip(policies, ys):
                y = torch.tensor([y], dtype=torch.long, device=device)
                loss += torch.nn.functional.cross_entropy(policy, y)
                y_pred = torch.argmax(policy)

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
