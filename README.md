# 👗 Fashion MNIST Classifier

This is a **Streamlit web app** that allows users to upload an image of a fashion item and predicts its class using a **VGG16 model trained on the Fashion MNIST dataset**. 

## 🚀 Features
- Upload an image of a clothing item
- Predicts the class with confidence
- Optionally shows a probability bar chart for all classes
- Interactive and visually appealing sidebar with instructions, components, and about sections

## 🏷️ Classes
1. T-shirt/top  
2. Trouser  
3. Pullover  
4. Dress  
5. Coat  
6. Sandal  
7. Shirt  
8. Sneaker  
9. Bag  
10. Ankle Boot  

## 📂 Folder Structure
fashion_mnist_streamlit/
│
├─ vgg16_weights.pth       # Trained PyTorch model  
├─ app4.py                  # Streamlit app code  
├─ README.md               # Project description  
├─ requirements.txt        # Dependencies  
└─ images/                 # (Optional) Example images  

## ⚡ How to Run
# 1. Clone the repository
git clone https://github.com/Hamzuuu200/fashion_mnist_streamlit.git  

# 2. Navigate into the folder
cd fashion_mnist_streamlit  

# 3. Install dependencies
pip install -r requirements.txt  

# 4. Run the app
streamlit run app4.py  

## 📦 Dependencies
streamlit  
torch  
torchvision  
numpy  
pillow  
plotly  

## 👨‍💻 Author
Muhammad Hamza
