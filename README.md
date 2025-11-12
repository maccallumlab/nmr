# nmr

NMR Chemical Shift Assignment using Graph Neural Networks and Reinforcement Learning

## Quick Start

### Generating Training Data

Create synthetic datasets for supervised pre-training:

```bash
# Generate a dataset
python scripts/generate_dataset.py --num-resid 10 --output dataset_10.pkl

# Generate training histories
python scripts/generate_histories.py \
    --dataset dataset_10.pkl \
    --num-histories 100
```

For detailed documentation, see [docs/fake-data-guide.md](docs/fake-data-guide.md)

## Documentation

- **[Fake Data Generation Guide](docs/fake-data-guide.md)** - Complete guide to generating synthetic training data
- **[Development Guide](docs/development-guide.md)** - Setup, development patterns, and testing
- **[Architecture Documentation](docs/architecture.md)** - System architecture and design
- **[Project Overview](docs/project-overview.md)** - Executive summary

## Project Structure

```
nmr/
├── nmr/                    # Main package
│   ├── construct.py        # Graph construction
│   ├── models/             # GNN architecture
│   └── env/                # RL environment
├── scripts/                # Executable scripts
│   ├── generate_dataset.py       # Dataset generation
│   ├── generate_histories.py     # History generation
│   ├── train_gnn.py              # Training loop
│   └── visualize_data.py         # Data visualization
├── tests/                  # Unit tests
└── docs/                   # Documentation
```
