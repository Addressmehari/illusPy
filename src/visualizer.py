import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

from src.model import ShapeCNN

def gaussian_blur_tensor(tensor, kernel_size=3, sigma=0.4):
    """Applies mild Gaussian blur to keep generated image crisp and smooth."""
    x = torch.arange(kernel_size) - kernel_size // 2
    grid = x ** 2
    gaussian_kernel = torch.exp(-grid / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    kernel2d = gaussian_kernel.unsqueeze(0) * gaussian_kernel.unsqueeze(1)
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2
    return F.conv2d(tensor, kernel2d, padding=padding)

def run_visualization(shape="square", base_dir="dataset", models_dir="models", iterations=300, lr=0.03, save_gif=True):
    """
    Live Interactive Side-by-Side Visualizer:
    - Panel A: Cycles through all dataset samples for target shape
    - Panel B: Morphs static noise into the CNN's learned mental image
    """
    model_path = os.path.join(models_dir, f"{shape}_cnn.pth")
    if not os.path.exists(model_path):
        model_path = f"{shape}_cnn.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    pos_dir = os.path.join(base_dir, f"{shape}s")
    sample_files = sorted(glob.glob(os.path.join(pos_dir, "*.png")))
    if len(sample_files) == 0:
        print(f"Error: No sample images found in '{pos_dir}'!")
        return

    print(f"Loaded {len(sample_files)} sample images for shape '{shape}'!")
    sample_images = [Image.open(fpath).convert("L") for fpath in sample_files]
    total_samples = len(sample_images)

    model = ShapeCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    input_img = (torch.randn(1, 1, 128, 128) * 0.15 + 0.5).clamp(0, 1)
    input_img.requires_grad = True

    plt.ion()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5))
    fig.canvas.manager.set_window_title(f"Live CNN Feature Visualization: {shape.upper()}")

    im_a = ax_a.imshow(sample_images[0], cmap='gray')
    title_a = ax_a.set_title(f"A: Dataset Sample #1/{total_samples}\n[Messy {shape.capitalize()} Flashcard]", fontsize=11, fontweight='bold')
    ax_a.axis('off')

    initial_np = input_img.detach().squeeze().numpy()
    im_b = ax_b.imshow(initial_np, cmap='inferno', vmin=0, vmax=1)
    title_b = ax_b.set_title(f"B: Learnt {shape.capitalize()} from CNN\nStep: 0/{iterations} | Score: 0.0%", fontsize=11, fontweight='bold')
    ax_b.axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    frames = []
    gif_dir = "visualizations"
    os.makedirs(gif_dir, exist_ok=True)
    steps_per_sample = max(1, iterations // total_samples)

    print(f"\n==================================================================")
    print(f"   STARTING LIVE VISUALIZATION FOR '{shape.upper()}' (All Samples & Steps) ")
    print(f"==================================================================")

    try:
        for step in range(1, iterations + 1):
            if not plt.fignum_exists(fig.number):
                print("\nPlot window closed by user.")
                break

            if input_img.grad is not None:
                input_img.grad.zero_()

            output_logit = model(input_img)
            prob_score = torch.sigmoid(output_logit)

            output_logit.backward()

            with torch.no_grad():
                grad = input_img.grad.data
                grad_std = grad.std()
                if grad_std > 1e-8:
                    grad = grad / grad_std

                input_img.data += lr * grad

                if step % 4 == 0:
                    input_img.data = gaussian_blur_tensor(input_img.data, kernel_size=3, sigma=0.4)
                input_img.data.clamp_(0, 1)

            sample_idx = min(total_samples - 1, (step - 1) // steps_per_sample)
            im_a.set_data(sample_images[sample_idx])
            title_a.set_text(f"A: Dataset Sample #{sample_idx+1}/{total_samples}\n[Messy {shape.capitalize()} Flashcard]")

            current_np = input_img.detach().squeeze().numpy()
            im_b.set_data(current_np)
            score_pct = prob_score.item() * 100
            title_b.set_text(f"B: Learnt {shape.capitalize()} from CNN\nStep: {step}/{iterations} | Score: {score_pct:.1f}%")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)

            if save_gif and (step % 3 == 0 or step == 1 or step == iterations):
                rgba = np.asarray(fig.canvas.buffer_rgba())
                frame_img = Image.fromarray(rgba[:, :, :3]).resize((1100, 500))
                frames.append(np.array(frame_img))

            if step % 30 == 0 or step == 1:
                print(f"Step [{step:03d}/{iterations}] | Panel A: Sample #{sample_idx+1}/{total_samples} | Score: {score_pct:.2f}%")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        final_img_np = input_img.detach().squeeze().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(final_img_np, cmap='inferno')
        plt.title(f"Generated Image from {shape.capitalize()} CNN\n(CNN's Learned Concept of a {shape.capitalize()})", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(gif_dir, f"cnn_generated_{shape}.png"), dpi=150)
        plt.close()

        if save_gif and len(frames) > 0:
            gif_name = "live_learning.gif" if shape == "square" else f"{shape}_live_learning.gif"
            gif_path = os.path.join(gif_dir, gif_name)
            print(f"\nSaving smooth animated GIF to '{gif_path}'...")
            imageio.mimsave(gif_path, frames, fps=15)
            print(f"GIF saved successfully!")

        print(f"\nLive visualization finished cleanly.")
        plt.ioff()
        plt.close('all')
