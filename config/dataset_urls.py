"""
Dataset URLs and Download Configuration
All datasets from Kaggle, Roboflow, and Zenodo
"""

DATASET_CONFIG = {
    # Kaggle Datasets
    'kaggle': {
        'sugarcane': {
            'name': 'akilesh253/sugarcane-plant-diseases-dataset',
            'type': 'kaggle',
            'diseases': ['healthy', 'mosaic', 'red_rot', 'rust', 'yellow_leaf']
        },
        'maize': {
            'name': 'smaranjitghose/corn-or-maize-leaf-disease-dataset',
            'type': 'kaggle',
            'diseases': ['healthy', 'blight', 'common_rust', 'gray_leaf_spot']
        },
        'wheat': {
            'name': 'kushagra3204/wheat-plant-diseases',
            'type': 'kaggle',
            'diseases': ['healthy', 'brown_rust', 'yellow_rust', 'septoria']
        },
        'ragi': {
            'name': 'prajwalbax/finger-millet-ragi-dataset',
            'type': 'kaggle',
            'diseases': ['healthy', 'blast', 'brown_spot']
        },
        'cotton': {
            'name': 'seroshkarim/cotton-leaf-disease-dataset',
            'type': 'kaggle',
            'diseases': ['healthy', 'bacterial_blight', 'curl_virus', 'fusarium_wilt']
        },
        'jute': {
            'name': 'mdsaimunalam/jute-leaf-disease-detection',
            'type': 'kaggle',
            'diseases': ['healthy', 'stemrot', 'anthracnose']
        },
        'pea': {
            'name': 'zunorain/pea-plant-dataset',
            'type': 'kaggle',
            'diseases': ['healthy', 'powdery_mildew', 'downy_mildew']
        },
        'rice': {
            'name': 'vbookshelf/rice-leaf-diseases',   # ✅ new rice dataset
            'type': 'kaggle',
            'diseases': ['bacterial_leaf_blight', 'brown_spot', 'leaf_smut']
        },
        'general_plants': {
            'name': 'vipoooool/new-plant-diseases-dataset',
            'type': 'kaggle',
            'description': 'Multi-crop disease dataset for additional training'
        }
    },

    # Roboflow Dataset
    'roboflow': {
        'bajra': {
            'workspace': 'leaf-chalmers',
            'project': 'pearl-millet-il2bp',
            'version': 1,
            'type': 'roboflow',
            'diseases': ['healthy', 'downy_mildew', 'blast']
        }
    },

    # Zenodo Dataset
    'zenodo': {
        'barley': {
            'record_id': '13734021',
            'type': 'zenodo',
            'diseases': ['healthy', 'net_blotch', 'scald', 'leaf_rust']
        }
    }
}

# Crop to Hindi name mapping
CROP_HINDI_NAMES = {
    'sugarcane': 'गन्ना',
    'maize': 'मक्का',
    'wheat': 'गेहूं',
    'bajra': 'बाजरा',
    'ragi': 'रागी',
    'cotton': 'कपास',
    'jute': 'जूट',
    'barley': 'जौ',
    'pea': 'मटर',
    'rice': 'चावल'  # ✅ added rice Hindi name
}
