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
$PYTHON_INTERPRETER create_cfg_exp6.py

#### Exp6 TCGA and MNIST ####
log_message "Starting Experiment 6: RNA and METH"
mkdir -p ./reports/paper-visualizations/Exp6
cp ./config_runs/Exp6/*.yaml .
make visualize RUN_ID=Exp6_TCGA_RNA_METH
log_message "Experiment 6 main run done"

log_message "Starting METH_METH run for comparison"
make visualize RUN_ID=Exp6_TCGA_METH_METH
log_message "Experiment 6 METH_to_METH run done"
# TODO new downstream scripts
