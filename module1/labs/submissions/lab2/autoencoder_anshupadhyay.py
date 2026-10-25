import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.ToTensor()

train_data = datasets.MNIST(root="data",train=True,download=True,transform=transform)

test_data = datasets.MNIST(root="data",train=False,download=True,transform=transform)

train_loader = DataLoader(train_data,batch_size=64,shuffle=True)

test_loader = DataLoader(test_data,batch_size=64,shuffle=False)


class encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # 28 x 28 -> 14 x 14
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),

            # 14 x 14 -> 7 x 7
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),

            # 7 x 7 -> 4 x 4
            # Bottleneck
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        )

    def forward(self, x):

        return self.network(x)

class decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # 4 x 4 -> 8 x 8
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1 ),
            nn.ReLU(),

            # 8 x 8 -> 16 x 16
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),

            # 16 x 16 -> 32 x 32
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)


class autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

model = autoencoder()

print(model)



opt = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


def train(model, images, opt):
    opt.zero_grad()

    reconstruction, z = model(images)

    reconstruction = reconstruction[:, :, 2:30, 2:30]
    loss = F.mse_loss( reconstruction,images)
    loss.backward()
    opt.step()
    return loss.item()


epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        loss = train(model,images,opt)

        total_loss += loss

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {average_loss:.4f}"
    )
model.eval()
test_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        reconstruction, z = model(images)
        reconstruction = reconstruction[:, :, 2:30, 2:30]
        loss = F.mse_loss(reconstruction,images)
        test_loss += loss.item()
test_loss = test_loss / len(test_loader)

print("\nTest MSE:", test_loss)
torch.save(model.state_dict(),"autoencoder.pth")

print("Model saved successfully!")