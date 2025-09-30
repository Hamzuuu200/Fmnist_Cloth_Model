from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt torch.manual_seed(42) from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv("imagedata.xlsx")
df.head() device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device} ") X = df.iloc[:,1:]. values
y = df.iloc[:,0] .values
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.2, random_state = 42) from torchvision.transforms import transforms
custom_transform = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485 , 0.456 , 0.406], std = [0.229 , 0.224 , 0.225])
])
from PIL  import Image
import numpy as np

class CustomDataset(Dataset):

  def __init__(self , features , labels , transform):
     self.features = features
     self.labels = labels
     self.transform = transform

  def __len__(self):
      return len(self.features)

  def __getitem__(self , index):

    image = self.features[index].reshape(28,28)
    image = image.astype(np.uint8)
    image = np.stack([image]*3 , axis = -1)
    image = Image.fromarray(image)
    image = self.transform(image)
    return image , torch.tensor(self.labels[index] , dtype = torch.long)
train_dataset = CustomDataset(X_train, y_train , transform = custom_transform)
test_dataset = CustomDataset(X_test , y_test , transform = custom_transform)
train_loader = DataLoader(train_dataset , batch_size = 32 ,shuffle = True)
test_loader  = DataLoader(test_dataset , batch_size = 32 , shuffle = False)import torchvision.models as models
vgg16 = models.vgg16(pretrained = True)
for param in vgg16.features.parameters():
   param.requires_grad = False
vgg16.classifier = nn.Sequential(
nn.Linear(25088 , 1024),
nn.ReLU(),
nn.Dropout(0.3),
nn.Linear(1024 , 512) ,
nn.ReLU(),
nn.Dropout(0.3),
nn.Linear(512 , 10)
)
vgg16 = vgg16.to(device) learning_rate = 0.0001
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_epoch_loss = 0

    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        outputs = vgg16(batch_features)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss / len(train_loader)
    print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")vgg16.eval()
total = 0
correct = 0
with torch.no_grad():
    for batch_features , batch_labels in test_loader:
        batch_features , batch_labels = batch_features.to(device) , batch_labels.to(device)
        outputs = vgg16(batch_features)
        _, predicted = torch.max(outputs , 1)
        loss = criterion(outputs , batch_labels)
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
        print(correct/total)
vgg16.eval()
train_total = 0
train_correct = 0
with torch.no_grad():
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = vgg16(batch_features)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()
train_accuracy = train_correct / train_total
print(f"Train Accuracy: {train_accuracy:.4f}")

test_total = 0
test_correct = 0
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = vgg16(batch_features)
        _, predicted = torch.max(outputs, 1)
        test_total += batch_labels.size(0)
        test_correct += (predicted == batch_labels).sum().item()
test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")
