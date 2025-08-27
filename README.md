🌱 Plant Disease Detection using CNN
📌 Overview

This project builds a Convolutional Neural Network (CNN) model to automatically detect plant diseases from images. It classifies plant leaves into categories (e.g., healthy, early blight, late blight, leaf mold, powdery mildew, septoria leaf spot), and also provides recommended solutions for each detected disease. The dataset is organized into subfolders for each class, and the model is trained using TensorFlow/Keras.

🎯 Objectives

Load and preprocess plant images from the dataset.

Train a CNN to classify diseases with high accuracy.

Evaluate model performance on a test set.

Allow users to upload an image and get:

Predicted disease

Suggested solution

🛠️ Tech Stack

Python 3.8+

TensorFlow / Keras – Model building and training

scikit-learn – Model evaluation

NumPy & Pandas – Data manipulation

Pillow (PIL) – Image preprocessing

Google Colab / Jupyter – Training environment

📂 Project Structure
├── plant_disease_model.keras      # Saved trained CNN model
├── fraud_detection.ipynb          # Example notebook (training workflow)
├── main.py                        # Python script for training & prediction
├── README.md                      # Documentation
└── dataset/                       # PlantVillage dataset (with subfolders per class)

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/yourusername/Plant-Disease-Detection.git
cd Plant-Disease-Detection


Install required packages:

pip install tensorflow numpy pillow scikit-learn


Download and organize the PlantVillage dataset.
Place it inside:

/content/drive/MyDrive/PlantVillage/
├── healthy/
├── early_blight/
├── late_blight/
├── leaf_mold/
├── powdery_mildew/
└── septoria_leaf_spot/

🚀 Usage
1️⃣ Training the Model

Run the main script:

python main.py


The model will load the dataset, preprocess images, train for a few epochs, and evaluate on the test set.

The trained model is saved as plant_disease_model.keras.

2️⃣ Making Predictions

After training, you can predict on new images:

Please enter the path to the plant image: path/to/image.jpg


Example output:

Disease: early_blight
Solution: Remove affected leaves, use a fungicide.

🧪 Model Architecture

Conv2D + MaxPooling2D layers (feature extraction)

Flatten + Dense layers (classification)

Dropout layer (to prevent overfitting)

Softmax activation for multi-class prediction

📊 Evaluation

Accuracy is computed on the test set using accuracy_score.

Prints overall Test Accuracy after training.

🌿 Example Diseases & Solutions

Healthy → No disease detected.

Early Blight → Remove affected leaves, use a fungicide.

Late Blight → Destroy infected plants, apply fungicide.

Leaf Mold → Improve air circulation, use fungicide.

Powdery Mildew → Apply sulfur-based fungicide.

Septoria Leaf Spot → Remove infected leaves, use fungicide.

📌 Future Improvements

Add more plant disease categories.

Use transfer learning (e.g., ResNet, EfficientNet) for higher accuracy.

Deploy as a web app or mobile app for real-time predictions.
