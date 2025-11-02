"""
Automated Dataset Downloader
Downloads from Kaggle, Roboflow, and Zenodo
"""
import os
import sys
import json
import time
import zipfile
import requests
import urllib3
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_urls import DATASET_CONFIG

# Disable SSL warnings for Zenodo
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DatasetDownloader:
    """Download datasets from multiple sources"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create download log
        self.log_file = self.output_dir.parent / 'download_log.json'
        self.download_log = self.load_log()
    
    def load_log(self):
        """Load download log"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_log(self):
        """Save download log"""
        with open(self.log_file, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_kaggle_dataset(self, dataset_name, crop_name):
        """Download dataset from Kaggle"""
        print(f"\n{'='*60}")
        print(f"Downloading Kaggle Dataset: {crop_name}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        crop_dir = self.output_dir / crop_name
        crop_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if crop_name in self.download_log:
            print(f"✓ {crop_name} already downloaded. Skipping...")
            return True
        
        try:
            # Import kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Authenticate
            api = KaggleApi()
            api.authenticate()
            
            print(f"Downloading to: {crop_dir}")
            
            # Download dataset
            api.dataset_download_files(
                dataset_name,
                path=crop_dir,
                unzip=True,
                quiet=False
            )
            
            # Log successful download
            self.download_log[crop_name] = {
                'source': 'kaggle',
                'dataset': dataset_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.save_log()
            
            print(f"✓ Successfully downloaded {crop_name}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading {crop_name}: {str(e)}")
            print(f"Please ensure:")
            print(f"  1. Kaggle API is installed: pip install kaggle")
            print(f"  2. API credentials are in ~/.kaggle/kaggle.json")
            print(f"  3. Dataset access is enabled")
            return False
    
    def download_roboflow_dataset(self, workspace, project, version, crop_name):
        """Download dataset from Roboflow"""
        print(f"\n{'='*60}")
        print(f"Downloading Roboflow Dataset: {crop_name}")
        print(f"Project: {workspace}/{project}")
        print(f"{'='*60}")
        
        crop_dir = self.output_dir / crop_name
        crop_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if crop_name in self.download_log:
            print(f"✓ {crop_name} already downloaded. Skipping...")
            return True
        
        try:
            from roboflow import Roboflow
            
            # Get API key from environment
            api_key = os.getenv('ROBOFLOW_API_KEY')
            if not api_key:
                print("⚠ ROBOFLOW_API_KEY not found in environment")
                print("Please set it in .env file or export it")
                print("Get your API key from: https://app.roboflow.com/settings/api")
                return False
            
            # Initialize Roboflow
            rf = Roboflow(api_key=api_key)
            
            # Get project
            proj = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download(
                model_format='folder',
                location=str(crop_dir)
            )
            
            # Log successful download
            self.download_log[crop_name] = {
                'source': 'roboflow',
                'workspace': workspace,
                'project': project,
                'version': version,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.save_log()
            
            print(f"✓ Successfully downloaded {crop_name}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading {crop_name}: {str(e)}")
            print(f"Manual download: https://universe.roboflow.com/{workspace}/{project}")
            return False
    
    def download_zenodo_dataset(self, record_id, crop_name):
        """Download dataset from Zenodo"""
        print(f"\n{'='*60}")
        print(f"Downloading Zenodo Dataset: {crop_name}")
        print(f"Record ID: {record_id}")
        print(f"{'='*60}")
        
        crop_dir = self.output_dir / crop_name
        crop_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if crop_name in self.download_log:
            print(f"✓ {crop_name} already downloaded. Skipping...")
            return True
        
        try:
            # Get record metadata
            api_url = f"https://zenodo.org/api/records/{record_id}"
            response = requests.get(api_url, verify=False)
            
            if response.status_code != 200:
                print(f"✗ Failed to fetch metadata: {response.status_code}")
                return False
            
            metadata = response.json()
            files = metadata.get('files', [])
            
            if not files:
                print(f"✗ No files found in record")
                return False
            
            print(f"Found {len(files)} files to download")
            
            # Download each file
            for file_info in tqdm(files, desc="Downloading files"):
                file_url = file_info['links']['self']
                file_name = file_info['key']
                file_path = crop_dir / file_name
                
                # Download file
                file_response = requests.get(file_url, stream=True, verify=False)
                
                if file_response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in file_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Unzip if it's a zip file
                    if file_name.endswith('.zip'):
                        print(f"Extracting {file_name}...")
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(crop_dir)
                        file_path.unlink()  # Remove zip file
                else:
                    print(f"✗ Failed to download {file_name}")
            
            # Log successful download
            self.download_log[crop_name] = {
                'source': 'zenodo',
                'record_id': record_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.save_log()
            
            print(f"✓ Successfully downloaded {crop_name}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading {crop_name}: {str(e)}")
            print(f"Manual download: https://zenodo.org/records/{record_id}")
            return False
    
    def download_all(self):
        """Download all datasets"""
        print("\n" + "="*60)
        print("AGRICULTURAL DISEASE DETECTION - DATASET DOWNLOADER")
        print("="*60)
        
        success_count = 0
        failed_count = 0
        
        # Download Kaggle datasets
        print("\n[1/3] Downloading Kaggle Datasets...")
        for crop_name, config in DATASET_CONFIG['kaggle'].items():
            if self.download_kaggle_dataset(config['name'], crop_name):
                success_count += 1
            else:
                failed_count += 1
            time.sleep(2)  # Rate limiting
        
        # Download Roboflow datasets
        print("\n[2/3] Downloading Roboflow Datasets...")
        for crop_name, config in DATASET_CONFIG['roboflow'].items():
            if self.download_roboflow_dataset(
                config['workspace'],
                config['project'],
                config['version'],
                crop_name
            ):
                success_count += 1
            else:
                failed_count += 1
            time.sleep(2)
        
        # Download Zenodo datasets
        print("\n[3/3] Downloading Zenodo Datasets...")
        for crop_name, config in DATASET_CONFIG['zenodo'].items():
            if self.download_zenodo_dataset(config['record_id'], crop_name):
                success_count += 1
            else:
                failed_count += 1
            time.sleep(2)
        
        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"✓ Successful: {success_count}")
        print(f"✗ Failed: {failed_count}")
        print(f"Total: {success_count + failed_count}")
        print("="*60)
        
        if failed_count > 0:
            print("\n⚠ Some downloads failed. Please:")
            print("1. Check your API credentials")
            print("2. Verify internet connection")
            print("3. Download failed datasets manually")


def main():
    """Main function"""
    print("Starting dataset download...")
    print("\nPrerequisites:")
    print("1. Kaggle API: pip install kaggle")
    print("2. Kaggle credentials in ~/.kaggle/kaggle.json")
    print("3. Roboflow: pip install roboflow")
    print("4. ROBOFLOW_API_KEY in environment\n")
    
    input("Press Enter to continue...")
    
    downloader = DatasetDownloader()
    downloader.download_all()
    
    print("\n✓ Download process completed!")
    print("Next step: Run 02_prepare_data.py to organize datasets")


if __name__ == '__main__':
    main()
