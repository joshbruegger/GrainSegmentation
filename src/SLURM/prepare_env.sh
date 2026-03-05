echo "Loading modules..."
module purge
module load Python/3.13.1-GCCcore-14.2.0
module list

export PATH="$HOME/.local/bin:$PATH"

uv sync