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
cuda_available=$($PYTHON_INTERPRETER -c "import torch; print(torch.cuda.is_available())")
log_message "CUDA available: $cuda_available"
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
cp ./reports/Exp4_Celegans_TF/figures/xmodal_vs_normal_test_boxplot.png ./reports/paper-visualizations/Exp4/Figure_S5_A.png
cp ./reports/Exp4_Celegans_TF/figures/translategrid_extra_class_labels.png ./reports/paper-visualizations/Exp4/Figure_4_H.png
cp ./reports/Exp4_Celegans_TF/figures/loss_plot_relative.png ./reports/paper-visualizations/Exp4/Figure_4_G.png
cp ./reports/Exp4_Celegans_TF/figures/xmodal_vs_normal_test_bar.png ./reports/paper-visualizations/Exp4/Figure_S5_B.png
cp ./reports/Exp4_Celegans_TF/xmodalix_eval_classifier_metrics.csv ./reports/paper-visualizations/Exp4/Table_S3.csv


# clean up
bash ./clean.sh Exp4_Celegans_TF,Exp4_CelegansImgImg true true # Clean up and keep only reports folder
rm ./Exp4_Celegans_TF_config.yaml
rm ./Exp4_Celegans_TFImgImg_config.yaml
log_message "Exp4 removed intermediate data"

log_message "Exp4 ALL DONE"
