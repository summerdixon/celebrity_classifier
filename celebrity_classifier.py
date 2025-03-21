import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

train_losses = []
val_accuracies = []

#verify GPU access
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

#params
batch_size = 32
img_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#preprocess the data
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#loader
dataset = datasets.ImageFolder(root='data', transform=transform)

#split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

#70% training
#20% validation
#10% test
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

#model def
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  #instead of pretrained=True
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

#loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training func
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=8):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss) #loss

        #validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = torch.sigmoid(model(images))
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100*correct/total
        val_accuracies.append(accuracy) #store accuracy
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

#training
train_model(model, train_loader, val_loader, criterion, optimizer)

#save
torch.save(model.state_dict(), 'celebrity_classifier.pth')
print("model saved!")

#graph for report viz
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Training Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', label='Validation Accuracy', color='red')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")  #image will be located in this folder
plt.show()