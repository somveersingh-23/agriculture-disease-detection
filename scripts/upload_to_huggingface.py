"""
Upload models to Hugging Face Hub
"""
from huggingface_hub import HfApi, create_repo
import os

# Login
os.system('huggingface-cli login')

# Create repository
api = HfApi()
repo_id ="somveersingh-23/agriculture-disease-detection"

try:
    create_repo(repo_id, repo_type="model")
except:
    print("Repo already exists")

# Upload models
api.upload_folder(
    folder_path="app/models",
    repo_id=repo_id,
    repo_type="model"
)

print(f"✓ Models uploaded to https://huggingface.co/{repo_id}")
