Seismic Noise Attenuation via Wavelet-Enhanced Self-Learning


This framework implements a self-supervised deep learning approach for seismic noise attenuation. It utilizes a noise-similarity pairing strategy combined with Discrete Wavelet Transform (DWT) domain learning to separate noise from effective signals without requiring clean ground-truth data.


1. Requirements & Dependencies

The code is built using PyTorch. We recommend using an Anaconda environment. Key dependencies include faiss (for efficient nearest neighbor search) and pytorch_wavelets.

# Basic dependencies
pip install torch torchvision numpy matplotlib scipy imageio lmdb tqdm

# Wavelet and Search dependencies
pip install pytorch-wavelets  # For DWTForward/DWTInverse
conda install -c pytorch faiss-gpu # For Faiss (GPU version recommended)

2. Usage Workflow
The training process consists of three sequential steps: Data Preparation, Configuration, and Model Training.



Step 1: Data Preparation (prepare_data.py)
This script constructs the self-supervised training dataset. It employs a block-matching algorithm (via Faiss) to find similar patches within the seismic volume and generates "noise-similarity" pairs stored in LMDB format for efficient I/O.

Key Parameters:

--data_folder: Path to your raw .npy seismic data.

--output_folder: Destination for the generated LMDB dataset.

--num_sim: Number of similar patches to select (default: 3,5,7).

--patch_size: Size of the patch for similarity search (default: 5).

Command:

python prepare_data.py 

Output: This will generate a folder ending in _lmdb and a corresponding .pkl meta-info file in the output directory.



Step 2: Configuration (seg512_factgus_1111_l1.py)
Before training, you must update the configuration file to point to the dataset generated in Step 1.

Open seg512_factgus_1111_l1.py.

Locate the data_train dictionary.

Update lmdb_file and meta_info_file with the absolute paths generated in Step 1.

Note: You can also adjust training hyperparameters here, such as epochs, batch_size, and gpu index.



Step 3: Training (train_dist1_DWT_4C.py)
The main training script implements a Wavelet-domain U-Net. It decomposes the input into four sub-bands (Approximation cA, Horizontal cH, Vertical cV, Diagonal cD) and applies targeted denoising strategies.

Command:

python train_dist1_DWT_4C.py --config_file seg512_factgus_1111_l1.py

3. Method Overview

Our method avoids the need for clean labels by exploiting the statistical independence between noise and signal across different scales.

Preprocessing: Raw seismic data is processed into noise-similarity pairs using Non-Local Means (NLM) block matching.

Network Architecture: A multi-branch U-Net architecture processes the DWT coefficients ($LL, LH, HL, HH$) separately.

Optimization: The network minimizes a hybrid loss function comprising pixel-wise reconstruction loss, structural similarity constraints, and wavelet-domain sparsity regularization.

