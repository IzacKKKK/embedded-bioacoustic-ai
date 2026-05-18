# Embedded Bioacoustic AI

Hardware-aware bioacoustic AI optimized for embedded inference.

---

## Overview

This repository contains the implementation and experimental work developed for a dissertation focused on efficient bioacoustic classification for embedded and edge AI systems.

The project investigates how lightweight and hardware-aware deep learning models can identify animal species from environmental audio recordings while maintaining high accuracy under constrained computational resources.

The main objective is to enable real-time bioacoustic monitoring on embedded and edge devices for biodiversity monitoring applications.

---

## Research Focus

This work explores several topics related to efficient deep learning and embedded AI, including:

- Bioacoustic classification
- Environmental audio analysis
- Embedded and Edge AI
- Hardware-aware optimization
- Lightweight neural network architectures
- Model compression techniques
- Quantization and pruning
- Real-time inference
- TinyML concepts
- Biodiversity monitoring systems

---

## Features

- Audio preprocessing pipeline
- Spectrogram generation
- Deep learning-based audio classification
- Lightweight and optimized neural networks
- Embedded-oriented inference optimization
- Evaluation and benchmarking tools
- Hardware-aware experimentation

---

## Key Results

- Up to **62× parameter reduction**
- More than **90% computational complexity reduction**
- Accuracy maintained above **92%**
- Optimized variants of ACDNet for embedded deployment
- Significant reduction in memory and computational requirements

---

## Repository Structure

```text
.
├── data/                  # Dataset references or sample data
├── docs/                  # Thesis notes and documentation
├── models/                # Model definitions and trained weights
├── notebooks/             # Experiments and exploratory analysis
├── results/               # Evaluation outputs and benchmarks
├── src/                   # Main source code
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignored files
└── README.md              # Project documentation
```

---

## Technologies

The project may include the following technologies and frameworks:

- Python
- PyTorch
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Audio signal processing techniques
- Embedded AI optimization methods

---

## Research Goals

The primary goals of this work are:

1. Develop efficient bioacoustic classification models
2. Reduce computational and memory requirements
3. Maintain high classification accuracy
4. Enable real-time inference on constrained devices
5. Support scalable biodiversity monitoring systems

---

## Future Work

Possible future directions include:

- Deployment on embedded hardware platforms
- FPGA acceleration
- Mobile application integration
- Real-time streaming inference
- Multi-species classification expansion
- Self-supervised audio representation learning

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/embedded-bioacoustic-ai.git
cd embedded-bioacoustic-ai
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Example:

```bash
python src/main.py
```

Additional experiments and evaluation scripts will be added later.

---

## Citation

If you use this work in research or academic projects, please cite the corresponding dissertation when available.

---

## License

This project is licensed under the MIT License.

---

## Author

Developed as part of a Master's dissertation in Computer Engineering and Intelligent Systems.
