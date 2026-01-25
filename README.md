EchoFind: Self-Supervised Audio Retrieval Engine
!!  Impulse 2026 Submission   !!!

## Project Overview
EchoFind is a self-supervised audio representation learning system designed to identify audio tracks in noisy, real-world environments. It leverages contrastive learning (SimCLR-style) to learn robust audio embeddings without requiring genre labels.

## 📂 Repository Structure
* `submission.py`: Core logic for the AudioEncoder and Retrieval System.
* `train.py`: Training loop implementing contrastive loss and augmentation.
* `generate_outputs.py`: Inference script for generating the submission CSV.
* `weights/`: Contains the pre-trained `encoder.pth` model weights.
* `notebooks/`: Visualization of the Latent Space embeddings.

##  How to Run !!!!
1. Install dependencies:
   ```bash
   pip install -r requirements.txt