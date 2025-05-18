# Gun Object Detection System with MLOps Integration

![Customs Gun Detection](https://img.shields.io/badge/Project-Gun%20Detection-red)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-blue)
![DVC](https://img.shields.io/badge/DVC-Integrated-green)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Monitoring-yellow)

## Project Overview

This project implements an advanced gun detection system leveraging deep learning and modern MLOps practices. It uses a Faster R-CNN model with ResNet-50 backbone to detect guns in images, and is built with a comprehensive MLOps pipeline for reproducibility and scalability.

The system is designed to help customs and security personnel identify potentially dangerous weapons in luggage scans or surveillance footage, enhancing security screening processes.

## Key Features

- **Deep Learning Model**: Implements Faster R-CNN with ResNet-50 backbone for accurate gun detection
- **Data Version Control**: Uses DVC for tracking datasets and model versions
- **API Integration**: FastAPI backend for real-time inference through REST API endpoints
- **Experiment Tracking**: TensorBoard integration for visualizing training metrics
- **Modular Architecture**: Clean separation of concerns for maintainable codebase
- **Containerization Ready**: Structured for easy Docker deployment
- **Exception Handling**: Custom exception framework for robust error tracking
- **Logging**: Comprehensive logging system for debugging and monitoring

## Technical Architecture

```
├── artifacts/           # Directory for models and data (DVC tracked)
│   ├── models/          # Trained models
│   ├── raw/             # Raw dataset
├── config/              # Configuration files
├── logs/                # Application logs
├── src/                 # Source code
│   ├── data_ingestion.py    # Data download and preparation 
│   ├── data_processing.py   # Dataset creation and transformations
│   ├── model_architecture.py # Model definition
│   ├── model_training.py    # Training pipeline
│   ├── custom_exception.py  # Exception handling
│   ├── logger.py            # Logging configuration
├── tensorboard_logs/    # TensorBoard logs
├── dvc.yaml             # DVC pipeline definition
├── main.py              # FastAPI application
├── requirements.txt     # Project dependencies
└── setup.py             # Package setup file
```

## MLOps Tools Integrated

- **DVC (Data Version Control)**: Manages data and model versions, enabling reproducible experiments
- **TensorBoard**: Visualizes training metrics and model performance
- **FastAPI**: Provides a modern, high-performance API framework for model serving
- **PyTorch**: Powers the deep learning model architecture and training
- **OpenCV**: Handles image processing and manipulation
- **Kaggle Hub**: Facilitates dataset downloads for training
- **Custom Logger**: Implements comprehensive application logging

## Pipeline Workflow

1. **Data Ingestion**: Downloads gun detection dataset from Kaggle
2. **Data Processing**: Creates PyTorch dataset with image-label pairs
3. **Model Training**: Trains Faster R-CNN on the prepared dataset
4. **Model Serving**: Exposes the model through FastAPI endpoints
5. **Monitoring**: Tracks model performance using TensorBoard

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.x
- CUDA-capable GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/customs-gun-object-detection.git 
cd customs-gun-object-detection

# Install dependencies
pip install -e .

# Initialize DVC
dvc init
```

### Running the Pipeline

```bash
# Pull the DVC-tracked data
dvc pull

# Run the full pipeline
dvc repro

# Start the API server
uvicorn main:app --reload
```

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/image.jpg"
```

## Model Training Details

- **Model**: Faster R-CNN with ResNet-50 backbone
- **Dataset**: Gun detection dataset from Kaggle
- **Training**: Using PyTorch with Adam optimizer
- **Metrics**: Model training tracked with TensorBoard
- **Output**: Bounding boxes around detected guns with confidence scores

## API Endpoints

- `GET /`: Welcome message
- `POST /predict/`: Upload an image for gun detection

## Future Improvements

- Integrate CI/CD pipeline with GitHub Actions
- Add model versioning with MLflow
- Implement A/B testing framework
- Containerize application with Docker and Kubernetes
- Add explainability features for model predictions
