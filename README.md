# AI-Powered Dermoscopic Skin Lesion Classification and Detection

## Overview
This project implements a deep learning-based system for classifying dermoscopic skin lesion images into benign and malignant categories. The objective is to assist in early detection of skin cancer using computer vision techniques.

The model is built using TensorFlow and Keras, based on a Convolutional Neural Network (CNN) trained on a labeled melanoma dataset.

## Dataset
- Name: Melanoma Skin Cancer Dataset  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset

### Structure
```
dataset/
│── train/
│   ├── Benign/
│   └── Malignant/
│── test/
    ├── Benign/
    └── Malignant/
```

## Tech Stack
- Python  
- TensorFlow, Keras  
- NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Google Colab  

## Methodology
- Image preprocessing: resizing (224×224), normalization  
- Data augmentation: rotation, shift, zoom, flip  
- CNN architecture with 3 convolutional layers + dense layers  
- Binary classification using sigmoid activation  

### Training
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Batch size: 32  
- Epochs: Up to 30 (EarlyStopping applied)  

## Results
- Training Accuracy: ~92–95%  
- Validation Accuracy: ~88–92%  
- Evaluation Metrics:
  - Precision, Recall, F1-score  
  - Confusion Matrix  

The model shows reliable performance in distinguishing between benign and malignant lesions with stable convergence.

## Deployment
The model can be deployed using a lightweight web interface.

- Suggested: Streamlit / Flask  
- Deployment Link: https://github.com/siddhi-works/Skin-Lesion-Classification-Using-Deep-Learning  

## Installation
```
pip install -r requirements.txt
```

## Usage
```
git clone https://github.com/siddhi-works/Skin-Lesion-Classification-Using-Deep-Learning.git
cd Skin-Lesion-Classification-Using-Deep-Learning
python main.py
```

## Project Structure
```
│── data/
│── models/
│── skin_cancer.ipynb
│── main.py
│── requirements.txt
│── README.md
```

## Future Improvements
- Transfer learning (EfficientNet, ResNet)  
- Multi-class skin disease classification  
- Real-time and mobile deployment  
- Larger and more diverse datasets  

## Disclaimer
This project is for educational purposes only and is not a substitute for professional medical diagnosis.

## License
MIT License
