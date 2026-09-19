# CNN Feature Visualization Research Project 🧠🖼️

A lightweight, modular research codebase for visualizing internal representations of Convolutional Neural Networks (CNNs) trained on geometric shapes (Squares and Circles) and objects (Cats) via **Activation Maximization (Gradient Ascent)**.

---

## 📁 Repository Structure

```
paper/
├── setup_dataset.py               # Single command dataset generator (Squares, Circles, Cats)
├── main.py                        # Unified CLI Runner for training & visualization
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Ignores dataset images, model weights, & outputs
├── README.md                      # Project Overview & Setup Instructions
├── research_guide.md              # Research Paper Writing Guide
├── draw_cat.py                    # Standalone programmatic cat illustrator
│
└── src/                           # Source Code Modules
    ├── model.py                   # PyTorch CNN Architecture (ShapeCNN)
    ├── dataset.py                 # Parametric Dataset Generator (Pillow)
    ├── train.py                   # Model Training Loop (BCE Loss + Adam)
    ├── visualizer.py              # Live Side-by-Side Visualizer & GIF Recorder
    └── classifier.py              # Image Classifier Predictor
```

---

## 🚀 Do this after cloned

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets (1 Command)
Creates 100 Squares, 100 Circles, and 100 Cats + Control shapes:
```bash
python setup_dataset.py
```

### 3. Train Models & Run Live Visualizations
```bash
# Square CNN
python main.py --shape square --mode train
python main.py --shape square --mode visualize

# Circle CNN
python main.py --shape circle --mode train
python main.py --shape circle --mode visualize

# Cat CNN
python main.py --shape cat --mode train
python main.py --shape cat --mode visualize
```

---

## 🔒 Git Tracking Policy (`.gitignore`)
To keep the git repository lightweight and fast, generated dataset images (`dataset/`), trained model weights (`models/`), and rendered GIFs (`visualizations/`) are **ignored by git**. Running `python setup_dataset.py` recreates all datasets locally in seconds.
