"""
CineScope — Model Training and Evaluation Script
Run this script to train the model and evaluate it.
Usage: python scripts/train_and_evaluate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import DatasetLoader
from src.train_model import train_model


def main():
    # Load dataset
    loader = DatasetLoader()
    data = loader.load_all()

    # Train and evaluate
    results = train_model(
        movies_df=data["movies"],
        ratings_df=data["ratings"],
        save_artifacts=True
    )

    print("\n✓ Training and evaluation complete")
    print(f"  Accuracy:  {results['metrics']['accuracy']:.4f}")
    print(f"  Precision: {results['metrics']['precision']:.4f}")
    print(f"  Recall:    {results['metrics']['recall']:.4f}")
    print(f"  F1-Score:  {results['metrics']['f1_score']:.4f}")


if __name__ == "__main__":
    main()
