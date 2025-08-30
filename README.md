# Threat Detection in Crowded Environments using Machine Learning

## 📖 Abstract
This project implements a machine learning-based system for detecting potential threats in crowded public spaces. The system utilizes deep learning models to analyze and classify activities in real-time, enabling early threat detection and alert mechanisms. This solution aims to enhance public safety in areas such as airports, train stations, and public venues where rapid threat identification is crucial.

## 🗂️ Project Structure
threat-detection-project/
├── models/
│ ├── final_model.pth # Main trained PyTorch model
│ ├── final_scaler.joblib # Primary data scaler
│ └── feature_scaler.joblib # Additional feature scaler
├── notebooks/
│ ├── trainingv2.ipynb # Model training notebook
│ ├── modelv4.ipynb # Model development notebook
│ ├── testing.ipynb # Testing and evaluation
│ ├── testingv3.ipynb # Additional testing
│ └── pipeline.ipynb # End-to-end pipeline
├── label_mapping.joblib # Class label mappings
└── .gitignore # Git exclusion rules

## 🚀 Installation & Usage

### Prerequisites
```bash
# Install required dependencies
pip install torch torchvision scikit-learn joblib pandas numpy jupyter

import torch
import joblib

# Load the trained model
model = torch.load('models/final_model.pth')
model.eval()  # Set to evaluation mode

# Load preprocessing components
scaler = joblib.load('models/final_scaler.joblib')
feature_scaler = joblib.load('models/feature_scaler.joblib')
label_mapping = joblib.load('label_mapping.joblib')

# Example: Preprocess new data
# processed_data = scaler.transform(raw_data)
# predictions = model(processed_data)

📊 Development Notebooks

trainingv2.ipynb: Model training procedures and hyperparameter tuning
modelv4.ipynb: Model architecture development and experimentation
testing.ipynb: Model evaluation and performance metrics
testingv3.ipynb: Additional validation tests
pipeline.ipynb: Complete end-to-end processing pipeline
🧠 Model Details

Framework: PyTorch
Architecture: Deep Neural Network (specific details in notebooks)
Input: Processed feature vectors from sensor/camera data
Output: Threat classification probabilities
🔧 Data Preprocessing

The system uses multiple preprocessing steps:

Feature scaling using final_scaler.joblib
Additional feature engineering with feature_scaler.joblib
Label encoding through label_mapping.joblib
📈 Performance

Model performance metrics and evaluation results are available in the testing notebooks. The system demonstrates high accuracy in identifying potential threats while maintaining low false positive rates.

📝 License

This project is intended for research and educational purposes. Please contact the authors for licensing information.