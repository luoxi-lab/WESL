# Seismic Noise Attenuation via Wavelet-Enhanced Self-Learning

This framework implements a **self-supervised deep learning** approach for **seismic noise attenuation**. It utilizes a **noise-similarity pairing strategy** combined with **Discrete Wavelet Transform (DWT)** domain learning to separate noise from effective signals **without requiring clean ground-truth data**.

---

## Requirements & Dependencies

The code is built using **PyTorch**. We recommend using an **Anaconda/Conda environment**. Key dependencies include:

- **Faiss**: efficient nearest neighbor search
- **pytorch-wavelets**
 
### Basic dependencies

```bash
pip install torch torchvision numpy matplotlib scipy imageio lmdb tqdm
```

### Wavelet & similarity-search dependencies

```bash
pip install pytorch-wavelets  # For DWTForward/DWTInverse

# For Faiss (GPU version recommended)
conda install -c pytorch faiss-gpu
```
---

## Usage Workflow

The training process consists of **three sequential steps**:

1. **Data Preparation**
2. **Configuration**
3. **Model Training**

---


## Step 1: Data Preparation (`prepare_data.py`)

This script constructs the self-supervised training dataset. It employs a **block-matching algorithm (via Faiss)** to find similar patches within the seismic volume and generates **"noise-similarity" pairs** stored in **LMDB** format for efficient I/O.

### Key Parameters

- `--data_folder`: Path to your raw `.npy` seismic data
- `--output_folder`: Destination for the generated LMDB dataset
- `--num_sim`: Number of similar patches to select *(default: 3, 5, 7)*
- `--patch_size`: Patch size for similarity search *(default: 5)*

### Command

```bash
python prepare_data.py
```


### Output

This will generate:

- A folder ending with `*_lmdb`
- A corresponding `*.pkl` meta-info file

Both will be saved in the output directory.

---

## Step 2: Configuration (`seg512_factgus_1111_l1.py`)

Before training, update the configuration file to point to the dataset generated in **Step 1**.

### What to do

1. Open `seg512_factgus_1111_l1.py`
2. Locate the `data_train` dictionary
3. Update:
   - `lmdb_file`: absolute path to the generated `*_lmdb` folder
   - `meta_info_file`: absolute path to the generated `*.pkl` file


### Notes

You can also adjust training hyperparameters here, such as:

- `epochs`
- `batch_size`
- GPU index (`gpu`)

---


## Step 3: Training (`train_dist1_DWT_4C.py`)

The main training script implements a **Wavelet-domain U-Net**. It decomposes the input into **four sub-bands**:

- Approximation: **cA**
- Horizontal: **cH**
- Vertical: **cV**
- Diagonal: **cD**

and applies targeted denoising strategies.

### Command

```bash
python train_dist1_DWT_4C.py --config_file seg512_factgus_1111_l1.py
```

---

## Method Overview

Our method avoids the need for clean labels by exploiting the **statistical independence between noise and signal across different scales**.

- **Preprocessing**: Raw seismic data is processed into noise-similarity pairs using **Non-Local Means (NLM)** block matching.
- **Network Architecture**: A **multi-branch U-Net** processes the DWT coefficients (**LL, LH, HL, HH**) separately.
- **Optimization**: The network minimizes a **hybrid loss** comprising:
  - pixel-wise reconstruction loss  
  - structural similarity constraints  
  - wavelet-domain sparsity regularization  

---


