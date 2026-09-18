import argparse
import sys
import os

from src.dataset import generate_full_dataset
from src.train import train_shape_model
from src.visualizer import run_visualization
from src.classifier import classify_image

def main():
    parser = argparse.ArgumentParser(description="CNN Feature Visualization & Classification Runner")
    parser.add_argument("--shape", type=str, default="square", choices=["square", "circle", "cat"], help="Shape/Object to experiment with (default: square)")
    parser.add_argument("--mode", type=str, default="visualize", choices=["visualize", "train", "generate_dataset", "classify"], help="Mode of execution")
    parser.add_argument("--image", type=str, default=None, help="Path to image file for classification mode")
    parser.add_argument("--steps", type=int, default=300, help="Number of gradient ascent steps for visualization (default: 300)")

    args = parser.parse_args()

    if args.mode == "generate_dataset":
        generate_full_dataset(target_shape=args.shape)
    elif args.mode == "train":
        train_shape_model(shape=args.shape)
    elif args.mode == "visualize":
        run_visualization(shape=args.shape, iterations=args.steps)
    elif args.mode == "classify":
        if not args.image:
            test_img = f"dataset/{args.shape}s/{args.shape}_000.png"
            print(f"No --image specified. Using default sample image: {test_img}")
            args.image = test_img
        classify_image(image_path=args.image, shape=args.shape)

if __name__ == "__main__":
    main()
