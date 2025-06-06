#!/bin/bash
if command -v curl &> /dev/null; then
    echo "curl is installed"
    curl -LsSf https://astral.sh/uv/0.5.9/install.sh | sh
    
elif command -v wget &> /dev/null; then
    echo "wget is installed"
    wget -qO- https://astral.sh/uv/0.5.9/install.sh | sh
else
    echo "Please install curl or wget"
    exit 1
fi
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
    source venv-gallia/bin/activate 
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
log_message "Creating GIFs"
$PYTHON_INTERPRETER eval-xmodalix-scripts/generate_gif.py Exp5_TCGA_MNIST # Corrected to use $PYTHON_INTERPRETER
log_message "Creating GIFs done"
# clean up
bash ./clean.sh Exp5_TCGA_MNIST,Exp5_TCGA_MNISTImggImg true true # Clean up and keep only reports folder
rm ./Exp5_Exp5_TCGA_MNIST_config.yaml
rm ./Exp5_Exp5_TCGA_MNISTImgImg_config.yaml
log_message "Exp5 removed intermediate data"

# Get paper visualization
log_message "Copying visualizations to reports/paper-visualizations/Exp5"
mkdir -p ./reports/paper-visualizations/Exp5/temp
cp ./reports/paper-visualizations/Exp5/* ./reports/paper-visualizations/Exp5/temp
# move the below files from temp to the main folder
cp ./reports/paper-visualizations/Exp5/temp/xmodal_vs_normal_test_boxplot.png ./reports/paper-visualizations/Exp5/Figure_S5_C_MSE_Boxplot.png
cp ./reports/paper-visualizations/Exp5/temp/xmodal_vs_normal_test_bar.png ./reports/paper-visualizations/Exp5/Figure_S5_D_MSE_Barplot.png
cp ./reports/paper-visualizations/Exp5/temp/xmodalix_eval_classifier_metrics.csv ./reports/paper-visualizations/Exp5/Table_S5_F1_Classifier.csv
cp ./reports/Exp5_TCGA_MNIST/figures/latent2D_Aligned_extra_class_labels.png ./reports/paper-visualizations/Exp5/Figure_4B_2DLatent.png
cp ./reports/Exp5_TCGA_MNIST/figures/loss_plot_relative.png ./reports/paper-visualizations/Exp5/Figure_4C_rel_loss.png
cp ./reports/Exp5_TCGA_MNIST/figures/translategrid_extra_class_labels.png ./reports/paper-visualizations/Exp5/Figure_4D_Digit_Grid.png
rm -r ./reports/paper-visualizations/Exp5/temp


log_message "Exp5 ALL DONE"
# rm -r ./reports/paper-visualizations/Exp5/temp
