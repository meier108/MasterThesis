"""Command line entry point for GB1 dataset generation.

Usage:
    python create_dataset.py --size 100000 --path results/gb1_generated.csv
"""

from data.gb1_data import main


if __name__ == "__main__":
    main()