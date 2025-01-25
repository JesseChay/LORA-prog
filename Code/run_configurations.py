import subprocess
import time
import os
from queue import Queue
from threading import Thread
import csv
from collections import OrderedDict

# Define experiment configurations
experiments = [
    # 1. Backbone comparison
    "--model_name resnet50 --finetune all --num_negatives 1 --loss infonce --temperature 0.07",
    "--model_name resnet18 --finetune all --num_negatives 1 --loss infonce --temperature 0.07",
    "--model_name efficientnet_b0 --finetune all --num_negatives 1 --loss infonce --temperature 0.07",
    
    # 2. Finetuning strategy comparison
    "--model_name resnet50 --finetune all --num_negatives 1 --loss infonce --temperature 0.07",
    "--model_name resnet50 --finetune head --num_negatives 1 --loss infonce --temperature 0.07",
    
    # 3. Number of negative samples
    "--model_name resnet50 --finetune all --num_negatives 1 --loss infonce --temperature 0.07",
    "--model_name resnet50 --finetune all --num_negatives 3 --loss infonce --temperature 0.07",
    "--model_name resnet50 --finetune all --num_negatives 5 --loss infonce --temperature 0.07",
    
    # 4. Loss function comparison
    "--model_name resnet50 --finetune all --num_negatives 1 --loss infonce --temperature 0.07",
    "--model_name resnet50 --finetune all --num_negatives 1 --loss triplet --margin 1.0",
    
    # 5. Temperature parameter study
    "--model_name resnet50 --finetune all --num_negatives 1 --loss infonce --temperature 0.05",
    "--model_name resnet50 --finetune all --num_negatives 1 --loss infonce --temperature 0.07",
    "--model_name resnet50 --finetune all --num_negatives 1 --loss infonce --temperature 0.1",
]

# Common arguments for all experiments
common_args = "--data_csv  /home/san/Projects/Packages/LORA_B5357/crawler/dataset_preparation_results.csv --batch_size 64  --epochs 5 --lr 1e-4 --seed 42"

# Create a queue of experiments and a dictionary to store experiment settings
experiment_queue = Queue()
experiment_settings = OrderedDict()

for i, exp in enumerate(experiments):
    output_dir = f"output/output_exp_{i}"
    experiment_queue.put((i, exp, output_dir))
    experiment_settings[i] = {"args": exp, "output_dir": output_dir}

# Function to save experiment settings to CSV
def save_experiment_settings(settings, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Experiment ID', 'Arguments', 'Output Directory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for exp_id, data in settings.items():
            writer.writerow({
                'Experiment ID': exp_id,
                'Arguments': data['args'],
                'Output Directory': data['output_dir']
            })

# Save experiment settings at the start
save_experiment_settings(experiment_settings, 'experiment_settings.csv')

# Function to run experiments on a GPU
def run_experiments_on_gpu(gpu_id):
    while not experiment_queue.empty():
        try:
            exp_id, exp_args, output_dir = experiment_queue.get(block=False)
        except Queue.Empty:
            break

        os.makedirs(output_dir, exist_ok=True)

        full_command = f"CUDA_VISIBLE_DEVICES={gpu_id} python src/main.py {common_args} {exp_args} --output_dir {output_dir}"
        
        print(f"Starting experiment {exp_id} on GPU {gpu_id}")
        print(f"Experiment settings: {exp_args}")
        print(f"Output directory: {output_dir}")
        
        subprocess.run(full_command, shell=True)
        print(f"Finished experiment {exp_id} on GPU {gpu_id}")

# Create and start a thread for each GPU
gpu_threads = []
for gpu_id in range(7):  # Assuming 7 GPUs (0-6)
    thread = Thread(target=run_experiments_on_gpu, args=(gpu_id,))
    thread.start()
    gpu_threads.append(thread)

# Wait for all threads to complete
for thread in gpu_threads:
    thread.join()

print("All experiments completed!")

# Print out all experiment settings and their corresponding output directories
print("\nExperiment Settings and Output Directories:")
for exp_id, data in experiment_settings.items():
    print(f"Experiment {exp_id}:")
    print(f"  Arguments: {data['args']}")
    print(f"  Output Directory: {data['output_dir']}")
    print()

print(f"A detailed CSV of all experiment settings has been saved to 'experiment_settings.csv'")
