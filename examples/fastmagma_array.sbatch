#!/bin/bash
#SBATCH --job-name=fastmagma
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=1-22
#SBATCH --output=fastmagma_%A_%a.log
# Activate an environment containing fastmagma before submitting this job.
# Export BFILE_PREFIX, ANNOT, PVAL_DIR, TRAITS, and OUT_DIR before sbatch.
set -euo pipefail
: "${BFILE_PREFIX:?Set BFILE_PREFIX to the PLINK prefix before chromosome number}"
: "${ANNOT:?Set ANNOT to the gene annotation file}"
: "${PVAL_DIR:?Set PVAL_DIR to the summary-statistics directory}"
: "${TRAITS:?Set TRAITS to a comma-separated trait list}"
: "${OUT_DIR:?Set OUT_DIR to the output directory}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
fastmagma run --chr "$SLURM_ARRAY_TASK_ID" \
  --bfile-prefix "$BFILE_PREFIX" --annot "$ANNOT" \
  --pval-dir "$PVAL_DIR" --traits "$TRAITS" --out-dir "$OUT_DIR" --threads "${SLURM_CPUS_PER_TASK:-4}"
