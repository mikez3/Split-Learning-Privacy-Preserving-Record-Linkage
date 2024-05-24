#!/bin/bash

# Define the configuration file
config_file="./prepare_job_config.sh"

# Define the datasets to be used in each iteration
datasets=(
    "./Data/part0_cosine_n500_ref200.csv"
    "./Data/part1_cosine_n500_ref200.csv"
    "./Data/part2_cosine_n500_ref200.csv"

    "./Data/part0_cosine_n500_ref2000.csv"
	"./Data/part1_cosine_n500_ref2000.csv"
    "./Data/part2_cosine_n500_ref2000.csv"

    "./Data/part0_cosine_n2000_ref200.csv"
    "./Data/part1_cosine_n2000_ref200.csv"
    "./Data/part2_cosine_n2000_ref200.csv"

    "./Data/part0_cosine_n2000_ref2000.csv"
    "./Data/part1_cosine_n2000_ref2000.csv"
    "./Data/part2_cosine_n2000_ref2000.csv"
)

# Define the source and destination directories for copying the file
# Adjust the kernel for linear:
model_file_site1="./workspace/simulate_job/app_site-1/model_local.joblib"
model_file_site2="./workspace/simulate_job/app_site-2/model_local.joblib"
model_destination_dir_local="./trained_models/shuffle/rbf/local/"
	
i=0
# Loop through each dataset
for dataset in "${datasets[@]}"; do
	echo "Processing dataset: $dataset"
	
	# Modify the DATASET_PATH variable in the config file
	sed -i "s|^DATASET_PATH=.*|DATASET_PATH=\"$dataset\"|" "$config_file"
	
	# Print the modified line to verify the change
    	grep "^DATASET_PATH=" "$config_file"
    	
    	 # Run the prepare_job_config.sh script
	bash "$config_file"
    	
    # Run the nvflare simulator command
	# Extract a portion of the dataset name to use in the new filename
	dataset_name=$(basename "$dataset")
   
    if [ $i -eq 0 ]; then
        echo "$dataset_name | $(date)"
    else
        echo -e "----------------------------------------------------------------------------------\n$dataset_name | $(date)"
    fi
	{ time nvflare simulator ./jobs/sklearn_svm_2_uniform -w ./workspace -n 2 -t 2 ; } 2>&1
	
	dataset_portion=${dataset_name::-4}
	
	# Define the new filename based on the dataset portion
	# Adjust the part number for every part
	model_new_filename_site1="site1_shuffle_part0_${dataset_portion}.joblib"
	model_new_filename_site2="site2_shuffle_part0_${dataset_portion}.joblib"

	# Copy the file, rename it, and paste it into another directory
	cp "$model_file_site1" "$model_destination_dir_local$model_new_filename_site1"
	cp "$model_file_site2" "$model_destination_dir_local$model_new_filename_site2"
	((i++))
done

