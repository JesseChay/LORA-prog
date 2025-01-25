
# Contrastive Learning for Image Similarity

This project implements contrastive learning techniques for image similarity tasks, specifically designed for LoRA (Low-Rank Adaptation) model images.

## Installation

### Step 1: Install PyTorch

Follow the official PyTorch installation guide: https://pytorch.org/get-started/locally/

For example, if you're using Linux with CUDA 12.4 and Conda, you can use:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Step 2: Install Dependencies

Install the required Python packages:

```
pip install argparse numpy pandas scikit-learn tqdm pillow
```

### Step 3: Optional - Jupyter Notebook Support

If you plan to use Jupyter notebooks:

```
pip install notebook matplotlib seaborn
conda install pytorch::faiss-gpu
```

## Usage

### Step 1: Prepare the Dataset

Download the zip file from the following google drive and uncompress it in project root directory:

https://drive.google.com/drive/folders/1f1oC7MIWTSeTVU3mCxXHDxlfJqL77U_V?usp=drive_link

```shell
unzip crawler.zip
```

The directory will show like this:
```shell
crawler/
src/
README.md
run_configurations.py
demo.ipynb
```

### Step 2: Train the Model

You can train the model using either the command line interface or a Jupyter notebook.

#### Command Line Interface

The CLI offers flexibility in setting various training parameters. Here's an example command:

```
python src/main.py --data_csv /path/to/dataset_preparation_results.csv --output_dir output/experiment_1 --epochs 20 --loss infonce --temperature 0.07 --num_negatives 1 --lr 1e-4
```

Key parameters:
- `--data_csv`: Path to your dataset CSV file
- `--output_dir`: Directory to save output files
- `--epochs`: Number of training epochs
- `--loss`: Loss function ('infonce' or 'triplet')
- `--temperature`: Temperature parameter for InfoNCE loss
- `--margin`: Margin parameter for Triplet loss
- `--num_negatives`: Number of negative samples per positive pair
- `--lr`: Learning rate

For a full list of parameters, run `python src/main.py --help`.

#### Jupyter Notebook

For a more interactive experience with visualizations, refer to `demo.ipynb`. Make sure you've completed the optional Step 3 in the installation process.

## Features

- Supports multiple backbone architectures (ResNet50, ResNet18, EfficientNet-B0)
- Implements both InfoNCE and Triplet loss functions
- Flexible data augmentation pipeline
- Customizable training configurations
- Hit rate evaluation metrics

## Output

The training process generates several output files in the specified output directory:
- `best_model.pth`: Weights of the best performing model
- `final_model.pth`: Weights of the model after the final epoch
- `training_history.csv`: Training and validation loss history
- `test_results.csv`: Test loss and hit rate metrics
