"""
Data Preparation and Organization
Organizes downloaded datasets for training

Added features:
 - optional downsampling of huge classes to reduce training time
 - automatic simple augmentation (oversampling) for small classes
 - goals are configurable via OVERSAMPLE_TARGETS and DOWNSAMPLE_MAX
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from PIL import Image, ImageOps, ImageEnhance

sys.path.append(str(Path(__file__).parent.parent))
from config.dataset_urls import DATASET_CONFIG, CROP_HINDI_NAMES


# -----------------------
# CONFIG: adjust these if you want
# -----------------------
# Target counts to oversample (create augmented images) for small classes
OVERSAMPLE_TARGETS = {
    'rice': 1000,   # rice: from 120 -> ~1000
    'jute': 1500,   # jute: from 920 -> 1500
    'ragi': 2000    # ragi: from 1514 -> 2000
}

# Maximum images to keep for very large classes (downsample to this)
DOWNSAMPLE_MAX = {
    'sugarcane': 10000,  # reduce 19k -> 10k
    'wheat': 10000       # reduce 14k -> 10k
}
# -----------------------

class DataPreparator:
    """Prepare and organize datasets"""

    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Create directories
        self.stage1_dir = self.processed_dir / 'stage1_crops'
        self.stage2_dir = self.processed_dir / 'stage2_diseases'

        for split in ['train', 'validation', 'test']:
            (self.stage1_dir / split).mkdir(parents=True, exist_ok=True)

        self.stats = defaultdict(lambda: defaultdict(int))

    def find_images(self, directory):
        """Find all images in directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []

        for ext in extensions:
            images.extend(directory.rglob(f'*{ext}'))

        # sort for reproducibility
        images = sorted(images)
        return images

    def split_data(self, images, train_ratio=0.8, val_ratio=0.1):
        """Split data into train/val/test"""
        random.shuffle(images)

        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train = images[:train_size]
        val = images[train_size:train_size + val_size]
        test = images[train_size + val_size:]

        return train, val, test

    def copy_images(self, image_list, dest_dir):
        """Copy images to destination"""
        dest_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(image_list):
            dest_path = dest_dir / f"{i:05d}_{img_path.name}"
            try:
                shutil.copy2(img_path, dest_path)
            except Exception:
                # if copy fails for any reason, skip
                continue

    # --- augmentation helpers ---
    def _augment_save(self, img_path, out_path, aug_index):
        """Create a few lightweight augmentations and save"""
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            return False

        # pick augmentation by aug_index (deterministic)
        w, h = img.size

        # basic transforms
        if aug_index % 4 == 0:
            # horizontal flip
            aug = ImageOps.mirror(img)
        elif aug_index % 4 == 1:
            # rotate small angle
            aug = img.rotate(15 if (aug_index % 2 == 0) else -15, expand=True).resize((w, h))
        elif aug_index % 4 == 2:
            # color jitter: increase contrast
            aug = ImageEnhance.Color(img).enhance(1.2)
        else:
            # slight crop + resize
            left = int(0.05 * w)
            upper = int(0.05 * h)
            right = int(w - 0.05 * w)
            lower = int(h - 0.05 * h)
            aug = img.crop((left, upper, right, lower)).resize((w, h))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            aug.save(out_path, format='JPEG', quality=88)
            return True
        except Exception:
            return False

    def _oversample_to_target(self, crop_dir, target_count):
        """
        Create augmented copies inside crop_dir/_aug to reach target_count.
        Returns list of all images (original + new aug).
        """
        crop_dir = Path(crop_dir)
        all_images = self.find_images(crop_dir)
        if not all_images:
            return []

        aug_dir = crop_dir / '_aug'
        aug_dir.mkdir(exist_ok=True)

        current = len(all_images)
        if current >= target_count:
            # include aug_dir files if any
            return self.find_images(crop_dir)  # original behavior

        idx = 0
        # iterate over originals cyclically and create augmentations
        originals = all_images.copy()
        while current < target_count:
            src = originals[idx % len(originals)]
            out_name = f"aug_{current:06d}.jpg"
            out_path = aug_dir / out_name
            success = self._augment_save(src, out_path, idx)
            if success:
                current += 1
            idx += 1
            # safety break
            if idx > target_count * 10:
                break

        return self.find_images(crop_dir)

    def _downsample_keep(self, images, keep_max):
        """
        Randomly downsample the list of image Paths to keep at most keep_max.
        """
        if len(images) <= keep_max:
            return images
        random.shuffle(images)
        kept = images[:keep_max]
        # return sorted for consistent behavior
        return sorted(kept)

    # --- main prepare functions ---
    def prepare_stage1_crop_classification(self):
        """
        Prepare Stage 1 data: Crop Classification
        Structure: stage1_crops/train/crop_name/images
        """
        print("\n" + "=" * 60)
        print("PREPARING STAGE 1: CROP CLASSIFICATION")
        print("=" * 60)

        all_crops = list(DATASET_CONFIG['kaggle'].keys())
        all_crops.extend(DATASET_CONFIG['roboflow'].keys())
        all_crops.extend(DATASET_CONFIG['zenodo'].keys())

        # Remove 'general_plants'
        if 'general_plants' in all_crops:
            all_crops.remove('general_plants')

        for crop in all_crops:
            crop_dir = self.raw_dir / crop

            if not crop_dir.exists():
                print(f"⚠ {crop} directory not found. Skipping...")
                continue

            print(f"\n Processing {crop} ({CROP_HINDI_NAMES.get(crop, crop)})...")

            # Find all images for this crop
            all_images = self.find_images(crop_dir)

            if not all_images:
                print(f"  ✗ No images found")
                continue

            print(f"  Found {len(all_images)} images (before adjustments)")

            # optional downsampling for very large classes
            if crop in DOWNSAMPLE_MAX:
                before = len(all_images)
                images_keep = self._downsample_keep(all_images, DOWNSAMPLE_MAX[crop])
                print(f"  Downsampling '{crop}' from {before} -> {len(images_keep)}")
                all_images = images_keep

            # optional oversampling (augmentation) for small classes
            if crop in OVERSAMPLE_TARGETS:
                target = OVERSAMPLE_TARGETS[crop]
                before = len(all_images)
                # create augmented images under raw_dir/<crop>/_aug
                all_images = self._oversample_to_target(crop_dir, target)
                after = len(all_images)
                print(f"  Oversampled '{crop}' from {before} -> {after} (target {target})")

            # Split data
            train, val, test = self.split_data(all_images)

            # Copy to Stage 1 directories
            self.copy_images(train, self.stage1_dir / 'train' / crop)
            self.copy_images(val, self.stage1_dir / 'validation' / crop)
            self.copy_images(test, self.stage1_dir / 'test' / crop)

            # Update stats
            self.stats['stage1'][crop] = {
                'train': len(train),
                'validation': len(val),
                'test': len(test),
                'total': len(all_images)
            }

            print(f"  ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def prepare_stage2_disease_detection(self):
        """
        Prepare Stage 2 data: Disease Detection
        Structure: stage2_diseases/crop_name/train/disease_name/images
        """
        print("\n" + "=" * 60)
        print("PREPARING STAGE 2: DISEASE DETECTION")
        print("=" * 60)

        # Combine all dataset sources
        all_datasets = {}
        all_datasets.update(DATASET_CONFIG['kaggle'])
        all_datasets.update(DATASET_CONFIG['roboflow'])
        all_datasets.update(DATASET_CONFIG['zenodo'])

        # Remove general dataset
        if 'general_plants' in all_datasets:
            del all_datasets['general_plants']

        for crop, config in all_datasets.items():
            crop_dir = self.raw_dir / crop

            if not crop_dir.exists():
                print(f"\n⚠ {crop} directory not found. Skipping...")
                continue

            print(f"\n Processing {crop} ({CROP_HINDI_NAMES.get(crop, crop)})...")

            # Find disease directories
            disease_dirs = [d for d in crop_dir.iterdir() if d.is_dir()]

            if not disease_dirs:
                # Try to find images directly
                all_images = self.find_images(crop_dir)
                if all_images:
                    # Assume single disease class
                    disease_name = 'unknown'
                    self._process_disease(crop, disease_name, all_images)
                continue

            # Process each disease
            for disease_dir in disease_dirs:
                disease_name = disease_dir.name.lower().replace(' ', '_').replace('-', '_')

                # Find images in this disease directory
                disease_images = self.find_images(disease_dir)

                if not disease_images:
                    continue

                self._process_disease(crop, disease_name, disease_images)

    def _process_disease(self, crop, disease_name, images):
        """Process a single disease for a crop"""
        print(f"  {disease_name}: {len(images)} images")

        if len(images) < 10:
            print(f"    ⚠ Too few images, skipping...")
            return

        # Split data
        train, val, test = self.split_data(images)

        # Create directories
        for split in ['train', 'validation', 'test']:
            (self.stage2_dir / crop / split).mkdir(parents=True, exist_ok=True)

        # Copy images
        self.copy_images(train, self.stage2_dir / crop / 'train' / disease_name)
        self.copy_images(val, self.stage2_dir / crop / 'validation' / disease_name)
        self.copy_images(test, self.stage2_dir / crop / 'test' / disease_name)

        # Update stats
        if crop not in self.stats['stage2']:
            self.stats['stage2'][crop] = {}

        self.stats['stage2'][crop][disease_name] = {
            'train': len(train),
            'validation': len(val),
            'test': len(test),
            'total': len(images)
        }

        print(f"    ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        # Stage 1 statistics
        print("\nStage 1: Crop Classification")
        print("-" * 60)
        total_stage1 = 0
        for crop, stats in self.stats['stage1'].items():
            total = stats['total']
            total_stage1 += total
            print(f"{crop:15} ({CROP_HINDI_NAMES.get(crop, crop):6}): {total:5} images")
        print(f"{'TOTAL':15} {'':6}: {total_stage1:5} images")

        # Stage 2 statistics
        print("\nStage 2: Disease Detection")
        print("-" * 60)
        for crop, diseases in self.stats['stage2'].items():
            print(f"\n{crop} ({CROP_HINDI_NAMES.get(crop, crop)}):")
            total_crop = 0
            for disease, stats in diseases.items():
                total = stats['total']
                total_crop += total
                print(f"  {disease:20}: {total:5} images")
            print(f"  {'Subtotal':20}: {total_crop:5} images")

    def prepare_all(self):
        """Execute full preparation pipeline"""
        print("\n" + "=" * 60)
        print("AGRICULTURAL DISEASE DETECTION")
        print("DATA PREPARATION PIPELINE")
        print("=" * 60)

        # Set random seed for reproducibility
        random.seed(42)

        # Prepare Stage 1
        self.prepare_stage1_crop_classification()

        # Prepare Stage 2
        self.prepare_stage2_disease_detection()

        # Print statistics
        self.print_statistics()

        print("\n" + "=" * 60)
        print("✓ DATA PREPARATION COMPLETED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 03_train_stage1.py to train crop classifier")
        print("2. Run 04_train_stage2.py to train disease models")


def main():
    """Main function"""
    preparator = DataPreparator()
    preparator.prepare_all()


if __name__ == '__main__':
    main()
