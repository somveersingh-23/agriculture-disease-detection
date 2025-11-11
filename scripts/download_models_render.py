"""
Download models on Render startup
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_all_models():
    """Download models from Hugging Face"""
    repo_id = os.getenv('somveersingh-23', 'somveersingh-23/agrofiesta_modelsDiseasedetection')
    
    print("Downloading models from Hugging Face...")
    
    # Download Stage 1
    print("Downloading Stage 1 model...")
    hf_hub_download(
        repo_id=repo_id,
        filename="stage1_crop_classifier.h5",
        local_dir="app/models",
        token=os.getenv('HF_TOKEN')
    )
    
    # Download Stage 2
    crops = ['sugarcane', 'maize', 'wheat', 'bajra', 'ragi', 
             'cotton', 'jute', 'barley', 'pea']
    
    for crop in crops:
        print(f"Downloading {crop} model...")
        hf_hub_download(
            repo_id=repo_id,
            filename=f"stage2_disease_models/{crop}_disease.h5",
            local_dir="app/models",
            token=os.getenv('HF_TOKEN')
        )
    
    print("✓ All models downloaded!")

if __name__ == '__main__':
    download_all_models()
