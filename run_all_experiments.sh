#!/bin/bash

PYTHON_INTERPRETER=python3
# Function to log messages with timestamp
log_message() {
    echo "[$(date +"%Y-%m-%d_%H-%M-%S")] $1"
}
# Check Python version
PYTHON_VERSION=$($PYTHON_INTERPRETER -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

# Compare the extracted version with 3.10
if [[ "$(printf '%s\n' "$PYTHON_VERSION" "3.10" | sort -V | head -n1)" != "3.10" ]]; then
    log_message "Error: Python 3.10 or higher is required. Current version is $PYTHON_VERSION. Change Line 3 in this file (run_all_experiments.sh) so that the selected interpreter is >=3.10"
    exit 1
fi

log_message "Python version $PYTHON_VERSION verified."

VENV_DIR="venv-gallia"
REQUIREMENTS_FILE="requirements.txt"

if [ -d "$VENV_DIR" ]; then
    log_message "Virtual environment '$VENV_DIR' found."

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"
    log_message "Virtual environment '$VENV_DIR' activated."
else
    log_message "Virtual environment '$VENV_DIR' does not exist. Creating..."
    make create_environment
    log_message "Activating environment"
    source venv-gallia/bin/activate ## TODO rename env name?
    log_message "Installing all requirements"
    make requirements
fi
# Check if PyTorch has CUDA available
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")

# If CUDA is available, export the CUBLAS_WORKSPACE_CONFIG variable
if [ "$cuda_available" == "True" ]; then
    export CUBLAS_WORKSPACE_CONFIG=:16:8
    log_message "CUDA is available. CUBLAS_WORKSPACE_CONFIG is set to :16:8."
else
    log_message "CUDA is not available. No changes made."
fi
####################################
#### Prepare Runs ##################
log_message "Getting preprocessed data"
bash ./get_preprocessed_data.sh

log_message "Creating configs"
$PYTHON_INTERPRETER create_cfg.py

mkdir -p ./reports/paper-visualizations

#### Exp1 Beta influence ###########
log_message "Starting Experiment 1: beta influence"

# copy cfg in root
cp ./config_runs/Exp1/Exp1_SC_Annealing_config.yaml .
cp ./config_runs/Exp1/Exp1_TCGA_Annealing_config.yaml .
# run AUTOENCODIX
make visualize RUN_ID=Exp1_SC_Annealing
make visualize RUN_ID=Exp1_TCGA_Annealing

# get paper visualization
mkdir -p ./reports/paper-visualizations/Exp1
python src/visualization/Exp1_visualization.py 

log_message "Cleaning up Experiment 1"
# clean up
bash ./clean.sh -r Exp1_SC_Annealing,Exp1_TCGA_Annealing -d -k # Clean up and keep only reports folder
rm ./Exp1_*_Annealing_config.yaml

log_message "Experiment 1 done"
###################################

#### Exp2 AE comparison ###########
log_message "Starting Experiment 2: AE comparison"
# copy cfg in root
cp ./config_runs/Exp2/*_config.yaml .

# run AUTOENCODIX
for config in ./Exp2*_config.yaml; do
    log_message "Current run: $(basename $config _config.yaml)"
    make ml_task RUN_ID=$(basename $config _config.yaml)
    bash ./clean.sh -r $(basename $config _config.yaml) -d -k # Clean up and keep only reports folder
done

# get paper visualization
mkdir -p ./reports/paper-visualizations/Exp2
$PYTHON_INTERPRETER ./src/visualization/Exp2_visualization.py

log_message "Cleaning up Experiment 2"
# clean up
rm ./Exp2*_config.yaml
log_message "Experiment 2 done"
###################################

#### Exp3 Ontix robustness ########
log_message "Starting Experiment 3: ontix robustness"
# copy cfg in root
cp ./config_runs/Exp3/*_config.yaml .

# run AUTOENCODIX
for config in ./Exp3*_config.yaml; do
    log_message "Current run: $(basename $config _config.yaml)"
    make ml_task RUN_ID=$(basename $config _config.yaml)       # make visualize might be sufficient
    bash ./clean.sh -r $(basename $config _config.yaml) -k -d # Clean up and keep only reports folder
done

# get paper visualization
mkdir -p ./reports/paper-visualizations/Exp3
$PYTHON_INTERPRETER ./src/visualization/Exp3_visualization.py

log_message "Cleaning up Experiment 3"
# clean up
rm ./Exp3*_config.yaml
log_message "Experiment 3 done"

#### Exp4 Celegans and TF Expression ####
log_message "Starting Experiment 4: Celegans and TF Expression"
mkdir -p ./reports/paper-visualizations/Exp4
cp ./config_runs/Exp4/*.yaml .

make visualize RUN_ID=Exp4_Celegans_TF
log_message "Experiment 4 main run done"

log_message "Starting Img_to_Img run for comparison"
make visualize RUN_ID=Exp4_Celegans_TFImgImg
log_message "Experiment 4 Img_to_Img run done"

log_message "Starting Celegans extra visualization"
log_message "Creating GIFs"
$PYTHON_INTERPRETER eval-xmodalix-scripts/generate_gif.py Exp4_Celegans_TF # Corrected to use $PYTHON_INTERPRETER
log_message "Creating GIFs done"

log_message "Creating comparison plots"
$PYTHON_INTERPRETER eval-xmodalix-scripts/plot_compare.py Exp4_Celegans_TF # Corrected to use $PYTHON_INTERPRETER
log_message "Creating comparison plots done"
log_message "Eval against Img_to_Img"
$PYTHON_INTERPRETER eval-xmodalix-scripts/eval_against_ImgImg.py Exp4_Celegans_TF # Corrected to use $PYTHON_INTERPRETER
log_message "Eval against Img_to_Img done"

log_message "Eval xmodalix with classification"
$PYTHON_INTERPRETER eval-xmodalix-scripts/eval_xmodalix.py Exp4_Celegans_TF # Corrected to use $PYTHON_INTERPRETER
log_message "Eval xmodalix with classification done"

log_message "Exp4 X-Modalix with regression"
$PYTHON_INTERPRETER eval-xmodalix-scripts/eval_xmodalix_regression.py Exp4_Celegans_TF # Corrected to use $PYTHON_INTERPRETER
log_message "Exp4 X-Modalix with regression done"

# Get paper visualization
log_message "Copying visualizations to reports/paper-visualizations/Exp4"
cp ./reports/Exp4_Celegans_TF/figures/* ./reports/paper-visualizations/Exp4
cp ./reports/Exp4_Celegans_TF/*.csv ./reports/paper-visualizations/Exp4 

# clean up
bash ./clean.sh -r Exp4_Celegans_TF,Exp4_CelegansImgImg -k -d # Clean up and keep only reports folder
rm ./Exp4_Celegans_TF_config.yaml
rm ./Exp4_Celegans_TFImgImg_config.yaml
log_message "Exp4 removed intermediate data"

log_message "Exp4 ALL DONE"


#### Exp5 TCGA and MNIST ####
log_message "Starting Experiment 5: TCGA and MNIST"
mkdir -p ./reports/paper-visualizations/Exp5
cp ./config_runs/Exp5/*.yaml .
make visualize RUN_ID=Exp5_TCGA_MNIST
log_message "Experiment 5 main run done"

log_message "Starting Img_to_Img run for comparison"
make visualize RUN_ID=Exp5_TCGA_MNISTImgImg
log_message "Experiment 5 Img_to_Img run done"

log_message "Starting TCGA extra visualization"

log_message "Creating comparison plots"
$PYTHON_INTERPRETER eval-xmodalix-scripts/plot_compare_tcga.py Exp5_TCGA_MNIST # Corrected to use $PYTHON_INTERPRETER
log_message "Creating comparison plots done"

log_message "Eval against Img_to_Img"
$PYTHON_INTERPRETER eval-xmodalix-scripts/eval_against_ImgImg.py Exp5_TCGA_MNIST # Corrected to use $PYTHON_INTERPRETER
log_message "Eval against Img_to_Img done"

log_message "Eval xmodalix with classification"
$PYTHON_INTERPRETER eval-xmodalix-scripts/eval_xmodalix.py Exp5_TCGA_MNIST # Corrected to use $PYTHON_INTERPRETER
log_message "Eval xmodalix with classification done"

#### SKIPPING THIS PART FOR TCGA BECAUSE WE HAVE NO TIMESEREIS DATA
og_message "Creating GIFs"
$PYTHON_INTERPRETER eval-xmodalix-scripts/generate_gif.py Exp5_TCGA_MNIST # Corrected to use $PYTHON_INTERPRETER
log_message "Creating GIFs done"
# clean up
bash ./clean.sh -r Exp5_TCGA_MNIST,Exp5_TCGA_MNISTImggImg -k -d # Clean up and keep only reports folder
rm ./Exp5_Exp5_TCGA_MNIST_config.yaml
rm ./Exp5_Exp5_TCGA_MNISTImgImg_config.yaml
log_message "Exp5 removed intermediate data"

log_message "Exp5 ALL DONE"
# Get paper visualization
log_message "Copying visualizations to reports/paper-visualizations/Exp4"
cp ./reports/Exp5_TCGA_MNIST/figures/* ./reports/paper-visualizations/Exp5
cp ./reports/Exp5_TCGA_MNIST/*.csv ./reports/paper-visualizations/Exp5