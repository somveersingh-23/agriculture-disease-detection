"""
Download models on Render startup (script run in build step)
This script uses huggingface_hub.hf_hub_download to pull files into app/models.
It will try a couple of Stage-1 filenames (mobilenet variant etc.) and download all stage2 models.
"""

import os
import sys
from huggingface_hub import hf_hub_download, RepositoryNotFoundError, RevisionNotFoundError

def download_all_models():
    """Download models from Hugging Face"""
    repo_id = os.getenv('HF_REPO_ID') or 'somveersingh-23/agriculture-disease-detection'
    hf_token = os.getenv('HF_TOKEN')  # if private repo, set in Render
    local_dir = "app/models"

    print(f"Downloading models from Hugging Face repo: {repo_id} -> {local_dir}")

    os.makedirs(local_dir, exist_ok=True)

    # Stage 1: try several candidate filenames
    stage1_candidates = [
        "stage1_crop_classifier.h5",
        "stage1_crop_classifier_mobilenetv2.h5",
        "stage1_crop_classifier_mobilenet_v2.h5",
    ]

    stage1_downloaded = False
    for fn in stage1_candidates:
        try:
            print(f"Trying to download Stage 1: {fn}")
            hf_hub_download(repo_id=repo_id, filename=fn, local_dir=local_dir, token=hf_token)
            # if successful, save to a stable name expected by app
            src = os.path.join(local_dir, fn)
            dst = os.path.join(local_dir, "stage1_crop_classifier.h5")
            if src != dst:
                try:
                    os.replace(src, dst)
                except Exception:
                    # fallback to copy if replace fails
                    import shutil
                    shutil.copyfile(src, dst)
            print(f"Downloaded Stage 1 as {dst}")
            stage1_downloaded = True
            break
        except (FileNotFoundError, RevisionNotFoundError, RepositoryNotFoundError) as e:
            print(f"Stage1 candidate not found: {fn} ({e})")
        except Exception as e:
            print(f"Failed to download {fn}: {e}")

    if not stage1_downloaded:
        print("Warning: Could not download any Stage 1 candidate. Please ensure model exists in the HF repo.")

    # Stage 2 crops (the files you uploaded)
    crops = ['bajra', 'cotton', 'jute', 'maize', 'pea', 'ragi', 'rice', 'sugarcane', 'wheat']

    for crop in crops:
        filename = f"stage2_disease_models/{crop}_disease.h5"
        try:
            print(f"Downloading {crop} model: {filename}")
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, token=hf_token)
            # hf_hub_download will save the file under local_dir/stage2_disease_models/<file>
            # move it to app/models/<crop>_disease.h5 (expected by the app)
            src = os.path.join(local_dir, "stage2_disease_models", f"{crop}_disease.h5")
            dst = os.path.join(local_dir, f"{crop}_disease.h5")
            if os.path.exists(src):
                try:
                    os.replace(src, dst)
                except Exception:
                    import shutil
                    shutil.copyfile(src, dst)
                # optionally remove the now-empty stage2_disease_models folder file entry if desired
            print(f"Downloaded {crop} model to {dst}")
        except (FileNotFoundError, RevisionNotFoundError, RepositoryNotFoundError) as e:
            print(f"{crop} model not found in repo: {filename} ({e})")
        except Exception as e:
            print(f"Failed to download {crop} model: {e}")

    print("✓ Model download script finished")


if __name__ == '__main__':
    download_all_models()
