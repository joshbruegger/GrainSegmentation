echo "Loading modules..."
module purge
module load cuDNN/9.10.1.4-CUDA-12.8.0
module load SciPy-bundle/2025.06-gfbf-2025a
module list

export PATH="$HOME/.local/bin:$PATH"

if [ ! -d ".venv" ]; then
    uv venv --system-site-packages
fi

uv sync