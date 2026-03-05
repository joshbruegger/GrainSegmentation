echo "Loading modules..."
module purge
module load Python/3.12.3-GCCcore-13.3.0
module list

echo "Preparing new job-specific environment..."

# ensure uv is in the path
export PATH="$HOME/.local/bin:$PATH"
echo "using uv: $(uv --version && which uv)"

# Tell uv to create and use the virtual environment in the node's local TMPDIR
export UV_PROJECT_ENVIRONMENT="$TMPDIR/.venv"
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

# Use copy link mode to avoid issues with hard links to the source filesystem
export UV_LINK_MODE=copy

# Ensure the pip-installed CUDA libraries are discoverable by TensorFlow
export LD_LIBRARY_PATH="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cudnn/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cublas/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cufft/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cusparse/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cusolver/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/nccl/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# Add nvcc to PATH so ptxas is available if TF needs it to JIT compile
export PATH="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"

# Tell XLA exactly where to find the libdevice file from the nvcc package
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvcc"
