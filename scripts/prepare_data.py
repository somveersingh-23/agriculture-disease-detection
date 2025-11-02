"""
Prepare datasets for training
Downloads from Kaggle and organizes into proper structure
"""
import os
import shutil
import zipfile
from pathlib import Path
import kaggle


class DatasetPreparator:
    """Prepare datasets for training"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_name, output_dir):
        """Download dataset from Kaggle"""
        print(f"Downloading {dataset_name}...")
        
        try:
            kaggle.api.dataset_download_files(
                dataset_name,
                path=output_dir,
                unzip=True
            )
            print(f"✓ Downloaded {dataset_name}")
        except Exception as e:
            print(f"✗ Error downloading {dataset_name}: {str(e)}")
    
    def download_all_datasets(self):
        """Download all required datasets"""
        
        datasets = {
            'sugarcane': 'akilesh253/sugarcane-plant-diseases-dataset',
            'maize': 'smaranjitghose/corn-or-maize-leaf-disease-dataset',
            'wheat': 'kushagra3204/wheat-plant-diseases',
            'ragi': 'prajwalbax/finger-millet-ragi-dataset',
            'cotton': 'seroshkarim/cotton-leaf-disease-dataset',
            'jute': 'mdsaimunalam/jute-leaf-disease-detection',
            'pea': 'zunorain/pea-plant-dataset',
            'general': 'vipoooool/new-plant-diseases-dataset'
        }
        
        for crop, dataset_name in datasets.items():
            output_dir = self.raw_dir / crop
            output_dir.mkdir(exist_ok=True)
            self.download_kaggle_dataset(dataset_name, output_dir)
    
    def organize_for_stage1(self):
        """
        Organize data for Stage 1 (Crop Classification)
        Structure: data/processed/stage1_crops/train/crop_name/images
        """
        print("\nOrganizing data for Stage 1 (Crop Classification)...")
        
        stage1_dir = self.processed_dir / 'stage1_crops'
        
        for split in ['train', 'validation', 'test']:
            split_dir = stage1_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each crop
        crops = ['sugarcane', 'maize', 'wheat', 'bajra', 'ragi', 
                 'cotton', 'jute', 'barley', 'pea']
        
        for crop in crops:
            crop_raw_dir = self.raw_dir / crop
            if not crop_raw_dir.exists():
                continue
            
            # Find all images
            image_files = list(crop_raw_dir.rglob('*.jpg')) + \
                         list(crop_raw_dir.rglob('*.png')) + \
                         list(crop_raw_dir.rglob('*.jpeg'))
            
            # Split data: 80% train, 10% val, 10% test
            total = len(image_files)
            train_size = int(0.8 * total)
            val_size = int(0.1 * total)
            
            train_files = image_files[:train_size]
            val_files = image_files[train_size:train_size + val_size]
            test_files = image_files[train_size + val_size:]
            
            # Copy files
            self._copy_files(train_files, stage1_dir / 'train' / crop)
            self._copy_files(val_files, stage1_dir / 'validation' / crop)
            self._copy_files(test_files, stage1_dir / 'test' / crop)
            
            print(f"✓ {crop}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    def organize_for_stage2(self):
        """
        Organize data for Stage 2 (Disease Detection)
        Structure: data/processed/stage2_diseases/crop_name/train/disease_name/images
        """
        print("\nOrganizing data for Stage 2 (Disease Detection)...")
        
        stage2_dir = self.processed_dir / 'stage2_diseases'
        
        # Each crop has its own disease classification
        crop_datasets = {
            'sugarcane': self.raw_dir / 'sugarcane',
            'maize': self.raw_dir / 'maize',
            'wheat': self.raw_dir / 'wheat',
            'bajra': self.raw_dir / 'bajra',
            'ragi': self.raw_dir / 'ragi',
            'cotton': self.raw_dir / 'cotton',
            'jute': self.raw_dir / 'jute',
            'pea': self.raw_dir / 'pea'
        }
        
        for crop, crop_dir in crop_datasets.items():
            if not crop_dir.exists():
                continue
            
            print(f"\nProcessing {crop}...")
            
            # Create crop directory with splits
            for split in ['train', 'validation', 'test']:
                (stage2_dir / crop / split).mkdir(parents=True, exist_ok=True)
            
            # Find disease directories
            disease_dirs = [d for d in crop_dir.iterdir() if d.is_dir()]
            
            for disease_dir in disease_dirs:
                disease_name = disease_dir.name.lower().replace(' ', '_')
                
                # Get all images
                image_files = list(disease_dir.rglob('*.jpg')) + \
                             list(disease_dir.rglob('*.png')) + \
                             list(disease_dir.rglob('*.jpeg'))
                
                if not image_files:
                    continue
                
                # Split data
                total = len(image_files)
                train_size = int(0.8 * total)
                val_size = int(0.1 * total)
                
                train_files = image_files[:train_size]
                val_files = image_files[train_size:train_size + val_size]
                test_files = image_files[train_size + val_size:]
                
                # Copy files
                self._copy_files(train_files, stage2_dir / crop / 'train' / disease_name)
                self._copy_files(val_files, stage2_dir / crop / 'validation' / disease_name)
                self._copy_files(test_files, stage2_dir / crop / 'test' / disease_name)
                
                print(f"  {disease_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    def _copy_files(self, file_list, destination):
        """Copy files to destination"""
        destination.mkdir(parents=True, exist_ok=True)
        
        for i, src_file in enumerate(file_list):
            dst_file = destination / f"{i:04d}_{src_file.name}"
            shutil.copy2(src_file, dst_file)
    
    def prepare_all(self):
        """Execute full preparation pipeline"""
        print("="*60)
        print("DATASET PREPARATION PIPELINE")
        print("="*60)
        
        # Step 1: Download datasets
        print("\n[1/3] Downloading datasets from Kaggle...")
        self.download_all_datasets()
        
        # Step 2: Organize for Stage 1
        print("\n[2/3] Organizing for Stage 1 (Crop Classification)...")
        self.organize_for_stage1()
        
        # Step 3: Organize for Stage 2
        print("\n[3/3] Organizing for Stage 2 (Disease Detection)...")
        self.organize_for_stage2()
        
        print("\n" + "="*60)
        print("DATASET PREPARATION COMPLETED!")
        print("="*60)


if __name__ == '__main__':
    preparator = DatasetPreparator()
    preparator.prepare_all()
