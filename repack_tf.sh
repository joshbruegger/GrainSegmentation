#!/bin/bash
#SBATCH --job-name=RepackTF
#SBATCH --output=logs/repack-tf-%j.log
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -e

# Set Apptainer directories to TMPDIR for fast local I/O
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
export APPTAINER_TMPDIR=$TMPDIR/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

echo "Moving to TMPDIR for fast local extraction and packing..."
cd $TMPDIR

echo "Pulling docker image and extracting TF..."
# We bind TMPDIR to /workspace so the wheel packs directly to the fast local disk
apptainer exec --bind $TMPDIR:/workspace docker://nvcr.io/nvidia/tensorflow:25.02-tf2-py3 bash -c "
  PYTHON_SITE=\$(python3 -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')
  echo \"Python site-packages at \$PYTHON_SITE\"
  
  mkdir -p /workspace/repack
  echo \"Copying tensorflow...\"
  cp -r \$PYTHON_SITE/tensorflow/ /workspace/repack/
  cp -r \$PYTHON_SITE/tensorflow-*/ /workspace/repack/
  
  echo \"Packing wheel...\"
  cd /workspace
  python3 -m wheel pack /workspace/repack --dest-dir /workspace
"

echo "Copying resulting wheel back to SCRATCH..."
mkdir -p $SCRATCH/GrainSeg/wheels
cp $TMPDIR/*.whl $SCRATCH/GrainSeg/wheels/

echo "Finished successfully! Wheel is at $SCRATCH/GrainSeg/wheels/"
