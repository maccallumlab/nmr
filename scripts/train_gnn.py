import torch
from pprint import pp
import sys
from pathlib import Path
import argparse

# Add parent directory to path to allow imports from nmr package
sys.path.insert(0, str(Path(__file__).parent.parent))

from nmr.construct import construct_graph
from nmr.models import NMRNet, ModelConfig
from nmr.nmr_gym.io import load_histories
from torch_geometric.loader import DataLoader


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


def preprocess_data(examples, device, config):
    """
    Convert state dictionaries to graphs with targets attached.

    Attaches action and value targets as graph-level attributes.
    This allows the DataLoader to handle graphs and targets together.

    Args:
        examples: List of (state_dict, action, value) tuples
        device: Device to place tensors on
        config: ModelConfig for graph construction

    Returns:
        List of HeteroData graphs with .action and .value attributes
    """
    graphs = []

    for state_dict, action, value in examples:
        graph = construct_graph(state_dict, device, config)

        # Attach targets as graph-level attributes (similar to shift_to_assign)
        graph.action = torch.tensor([action], dtype=torch.long, device=device)
        graph.value = torch.tensor([value], dtype=torch.float32, device=device)

        graphs.append(graph)

    return graphs


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN model on NMR assignment histories"
    )
    parser.add_argument(
        "--histories",
        type=str,
        required=True,
        help="Path to the pickle file containing training histories",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training (default: cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50000,
        help="Number of training epochs (default: 50000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate model every N iterations (default: 100)",
    )
    parser.add_argument(
        "--num-nmr-layers",
        type=int,
        default=1,
        help="Number of NMR layers in the model (default: 1)",
    )

    args = parser.parse_args()

    # Set device
    device = args.device

    # Create config (needed for both graph construction and network)
    config = ModelConfig(num_nmr_layers=args.num_nmr_layers)

    # Load data
    examples = extract_data(args.histories)
    nmr_graphs = preprocess_data(examples, device, config)

    batch_size = args.batch_size
    epochs = args.epochs
    eval_interval = args.eval_interval

    # Targets are now attached to graphs, so shuffling is safe
    data_loader = DataLoader(nmr_graphs, batch_size=batch_size, shuffle=False)

    # Create our network and optimizer
    net = NMRNet(device, config)
    opt = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=0.01)

    iteration = 0
    for epoch in range(epochs):

        for xs in data_loader:
            net.train()
            iteration += 1

            _, policies = net(xs)

            # Unbatch graphs to access per-graph attributes
            # Targets (action, value) are attached to each graph
            graphs_list = xs.to_data_list()

            loss = 0
            correct = 0
            count = 0
            policy_ce_losses = []

            # Policy training: Supervise the network to predict the correct residue assignment
            # - policy: shape [num_peaks, num_residues], where policy[peak_i, res_j] = logit for peak i -> residue j
            # - We compute cross-entropy loss for:
            #   1. The peak being assigned (shift_to_assign) with action as target
            #   2. All previously assigned peaks with their assigned residues as targets
            for graph, policy in zip(graphs_list, policies):
                action = graph.action  # Target residue for current assignment
                shift_to_assign = graph.shift_to_assign.item()

                # Cross-entropy loss for current assignment
                # Extract row for the peak being assigned: policy[shift_to_assign, :]
                current_logits = policy[shift_to_assign].unsqueeze(0)  # [1, num_residues]
                ce_loss = torch.nn.functional.cross_entropy(current_logits, action)

                # Cross-entropy loss for all previous assignments
                edge_index = graph["Peak", "assigned_to", "Residue"].edge_index
                if edge_index.shape[1] > 0:  # Check if there are any assigned peaks
                    assigned_peak_ids = edge_index[0]  # Indices of assigned peaks
                    assigned_residue_ids = edge_index[1]  # Indices of assigned residues

                    # Extract rows for all assigned peaks: policy[assigned_peak_ids, :]
                    # Shape: [num_assigned, num_residues]
                    assigned_logits = policy[assigned_peak_ids]

                    # Compute cross-entropy for each previous assignment and sum
                    # Target for each row is the corresponding assigned residue
                    prev_ce_loss = torch.nn.functional.cross_entropy(
                        assigned_logits, assigned_residue_ids, reduction='sum'
                    )
                    ce_loss = ce_loss + prev_ce_loss

                # Track components for logging
                policy_ce_losses.append(ce_loss.item())

                loss += ce_loss

                y_pred = torch.argmax(current_logits)
                if y_pred == action:
                    correct += 1
                count += 1

            if iteration % eval_interval == 0:
                mean_ce = sum(policy_ce_losses) / len(policy_ce_losses)
                print(
                    f"{iteration}, "
                    f"Policy Loss: {mean_ce:.4f}, "
                    f"Accuracy: {correct / count:.4f}"
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

if __name__ == "__main__":
    main()