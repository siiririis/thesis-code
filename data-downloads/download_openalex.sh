#!/bin/bash
#SBATCH --job-name=oa-new-query
#SBATCH --account=project_xxxxxxx
#SBATCH --output=outputs/%x_%j.out
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --gres=nvme:20

echo "Initializing python modules"
module load python-data
source /projappl/project_xxxxxxx/venv/bin/activate

# Create a folder in temporary local storage
WORKDIR=$LOCAL_SCRATCH/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p $WORKDIR
echo "Working directory: $WORKDIR"

# These are not needed with a fresh query
echo "Copying the checkpoint file into local storage"
cp /scratch/project_xxxxxxx/data_collection/data/.openalex-checkpoint.json $WORKDIR

# Check API key status (or not as it uses credits)
# openalex status --api-key INSERT_API_KEY

# Run download
openalex download \
    --api-key INSERT_API_KEY \
    --output $WORKDIR \
    --workers 20 \
    --filter 'title_and_abstract.search:"conversational agent" OR "agent system" OR ai OR "artificial intelligence" OR chatbot OR genai OR "language model",is_oa:true,type:article|preprint,has_abstract:true,language:en,has_doi:true' \
    --nested \
    --resume

echo "Archiving and compressing downloaded files, preserving the folder structure..."
tar -czf /scratch/project_xxxxxxx/data_collection/data/metadata-downloads/oa-new-query-02.tar.gz -C $LOCAL_SCRATCH ${SLURM_JOB_NAME}_${SLURM_JOB_ID}
echo "Copied to scratch."

# Where to save the JSONL file(s)
DEST=/scratch/project_xxxxxxx/data_collection/data/jsonl-data

echo "Creating JSONL file(s)..."
srun python3 json_to_jsonl.py $WORKDIR $DEST --filename oa-new-query

echo "Job completed!"

seff ${SLURM_JOB_ID}