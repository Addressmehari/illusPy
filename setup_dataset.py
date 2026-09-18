"""
Dataset Setup Script for CNN Feature Visualization Research.
Generates all training datasets:
  - Squares & Non-Squares (dataset/squares & dataset/non_squares)
  - Circles & Non-Circles (dataset/circles & dataset/non_circles)
  - Cats & Non-Cats       (dataset/cats & dataset/non_cats)

Run this script once after cloning the repo:
    python setup_dataset.py
"""
import sys
import os

from src.dataset import generate_full_dataset

def main():
    print("==================================================================")
    print("   Setting up Datasets for CNN Feature Visualization Research     ")
    print("==================================================================")
    
    shapes = ["square", "circle", "cat"]
    for shape in shapes:
        print(f"\n--- Generating {shape.upper()} & NON-{shape.upper()} Datasets ---")
        generate_full_dataset(target_shape=shape, base_dir="dataset", count=100)

    print("\n==================================================================")
    print("   [SUCCESS] ALL DATASETS (cats, non_cats, squares, circles) READY! ")
    print("==================================================================")
    print("\nGenerated folders in dataset/:")
    print("  - dataset/cats & dataset/non_cats")
    print("  - dataset/squares & dataset/non_squares")
    print("  - dataset/circles & dataset/non_circles")
    print("\nNext steps:")
    print("  1. Train models:       python main.py --shape cat --mode train")
    print("  2. Run visualization:  python main.py --shape cat --mode visualize")

if __name__ == "__main__":
    main()
