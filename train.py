import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from model_definition import FishDiseaseModel

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR="data/fishdiseasedataset"

train_dir=os.path.join(DATA_DIR,"Train")
test_dir=os.path.join(DATA_DIR,"Test")

train_transform=transforms.Compose([
	transforms.Resize((224,224)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(20),
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform=transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset=datasets.ImageFolder(train_dir,transform=train_transform)
test_dataset=datasets.ImageFolder(test_dir,transform=test_transform)

train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=16,shuffle=False)

class_names=train_dataset.classes
num_classes=len(class_names)

model=FishDiseaseModel(num_classes).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)

epochs=20
train_losses=[]

for epoch in range(epochs):
	model.train()
	running_loss=0.0
	for images,labels in train_loader:
		images,labels=images.to(device),labels.to(device)
		optimizer.zero_grad()
		outputs=model(images)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
	epoch_loss=running_loss/len(train_loader)
	train_losses.append(epoch_loss)
	print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(),"fish_disease_model.pth")
print("Model saved successfully!")

plt.plot(train_losses)
plt.show()