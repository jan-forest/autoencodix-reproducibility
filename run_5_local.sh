#!/bin/bash

PYTHON_INTERPRETER=python
# Function to log messages with timestamp
log_message() {
    echo "[$(date +"%Y-%m-%d_%H-%M-%S")] $1"
}

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
# cuda_available=$($PYTHON_INTERPRETER -c "import torch; print(torch.cuda.is_available())")
cude_available="False"

####################################
#### Prepare Runs ##################
log_message "Getting preprocessed data"
bash ./get_preprocessed_data.sh

log_message "Creating configs"
$PYTHON_INTERPRETER create_cfg.py

mkdir -p ./reports/paper-visualizations


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