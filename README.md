# 🌾 Agriculture Disease Detection API

**कृषि रोग पहचान API** - A comprehensive crop disease detection system designed specifically for Indian farmers.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Crops](#supported-crops)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Training Models](#training-models)
- [Deployment](#deployment)
- [API Usage](#api-usage)
- [Android Integration](#android-integration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🎯 Overview

This project provides a **two-stage hierarchical disease detection system** that:

1. **Stage 1**: Identifies the crop type from a leaf image (97%+ accuracy)
2. **Stage 2**: Detects diseases specific to that crop using specialized models
3. Provides **farmer-friendly treatment recommendations** in Hindi and English

The system is optimized for:
- ✅ **Small farmers** with limited resources
- ✅ **Backward areas** with low literacy rates
- ✅ **Simple language** instructions (recipe-style)
- ✅ **Both home remedies and chemical treatments**

## ✨ Features

### 🔬 Technical Features

- **Two-stage hierarchical architecture** for better accuracy
- **EfficientNetB0** backbone (97.10% accuracy on crop classification)
- **Separate disease models** for each crop (scalable design)
- **Real-time inference** optimized for mobile devices
- **Farmer-friendly JSON responses** with detailed treatment plans

### 🌾 Agricultural Features

- **9 Major Crops** supported initially
- **40+ Disease classes** across all crops
- **Multi-language support** (Hindi primary, English)
- **Home remedies** with local ingredients
- **Chemical treatments** for small and large fields
- **Cost estimates** for all treatments
- **Prevention tips** and expert contact info

### 🚀 Production Features

- **FastAPI** backend with async support
- **Docker containerization**
- **Render deployment** ready
- **Health checks** and monitoring
- **Comprehensive logging**
- **CORS enabled** for mobile apps

## 🌱 Supported Crops

| Crop | Hindi Name | Scientific Name | Diseases Supported |
|------|------------|-----------------|-------------------|
| Sugarcane | गन्ना | *Saccharum officinarum* | Mosaic, Red Rot, Rust, Yellow Leaf |
| Maize | मक्का | *Zea mays* | Blight, Common Rust, Gray Leaf Spot |
| Wheat | गेहूं | *Triticum aestivum* | Brown Rust, Yellow Rust, Septoria |
| Bajra | बाजरा | *Pennisetum glaucum* | Downy Mildew, Blast |
| Ragi | रागी | *Eleusine coracana* | Blast, Brown Spot |
| Cotton | कपास | *Gossypium* | Bacterial Blight, Curl Virus, Fusarium Wilt |
| Jute | जूट | *Corchorus* | Stem Rot, Anthracnose |
| Barley | जौ | *Hordeum vulgare* | Net Blotch, Scald, Leaf Rust |
| Pea | मटर | *Pisum sativum* | Powdery Mildew, Downy Mildew |

**Future expansion**: Home decor plants, gardening, vegetables

## 🏗️ System Architecture

