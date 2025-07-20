**🌿 Plant Disease Prediction using Deep Learning 🌱**

This project builds a Convolutional Neural Network (CNN) to classify plant leaves as **Healthy** or **Diseased** using image data. It includes both a model training script and a GUI for real-time predictions. This tool can assist in early detection and diagnosis of plant diseases to support smart farming and crop health monitoring.

---

**📂 Dataset**

📥 The dataset is **publicly available** on Kaggle here:  
🔗 [Plant Disease Prediction (Disease and Healthy)](https://www.kaggle.com/datasets/dittakavinikhita/plant-disease-prediction-disease-and-healthy)

### Download Instructions:

1. Go to the Kaggle dataset link above.
2. Click **Download** (top-right).
3. Unzip the downloaded file.
4. Place the extracted `Dataset/` folder in the root of this project directory.

The folder should contain:
Dataset/
├── Disease/
└── Healthy/

Each subfolder contains labeled plant leaf images.

---

## ⚙️ How to Run the Project

### 1. 📦 Install Dependencies

First, set up a virtual environment (optional but recommended):

```bash
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# or (macOS/Linux)
source venv/bin/activate

Then install required packages:
pip install -r requirements.txt

### 2. 🧠 Train the Model
Run the training script:
python train.py
This will:

Load and preprocess images

Train a CNN model (or load a saved model if it exists)

Save the model architecture and weights to the model/ folder

You’ll see output like training accuracy, model summary, and progress over 20 epochs.

3. 🖼 Predict Using GUI
Once the model is trained, run the GUI for real-time predictions:

python test_code.py
This will open a simple window where you can:

📁 Select a leaf image

🤖 Get a prediction: "Healthy" or "Disease"

🎯 See the confidence score

Make sure model/model.json and model/model.weights.h5 exist before running the GUI. Otherwise, train the model first.

🧠 Model Overview
CNN architecture using Keras with TensorFlow backend

Input image size: 32x32 RGB

2 Convolutional layers + MaxPooling

Flatten + Dense + Softmax

Categorical cross-entropy loss

Adam optimizer

📁 Project Structure
graphql
Copy
Edit
plant_disease/
├── Dataset/              # Folder with images (Healthy/ and Disease/)
├── model/                # Stores saved model weights, JSON, and training history
├── train.py              # Script to train the CNN model
├── test_code.py          # GUI for image classification
├── requirements.txt      # Python dependencies
└── README.md

🔐 License
This project is open-source and free to use for educational and non-commercial purposes.
If you use this work in a publication or application, please consider crediting the original dataset uploader.

🙏 Acknowledgments
📸 Kaggle Dataset by @dittakavinikhita

🧠 TensorFlow & Keras for deep learning

💡 Inspiration from plant pathology research
