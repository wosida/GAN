#深度卷积生成对抗网络
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.ConvTranspose2d(100,512,4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512,1024,4,2,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Tanh(),
        )
    def forward(self,x):
        return self.cnn(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,1,4), #(b,1,1,1)
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        return self.cnn(x).squeeze()

def train(generator,discriminator,epochs,trainloder,device,g_optimizer,d_optimizer):
    global d_loss, i, g_loss
    for i in range(epochs):
        for i,(img,_) in enumerate(trainloder):
            img=img.to(device)
            one_labels=torch.ones(img.size(0)).to(device)
            zero_labels=torch.zeros(img.size(0)).to(device)
            noise=torch.randn(img.size(0),100,1,1).to(device)
            #生成假图
            fake_img=generator(noise)
            fake_score=discriminator(fake_img)
            real_score=discriminator(img)

            d_loss=nn.BCELoss()(fake_score,zero_labels)+nn.BCELoss()(real_score,one_labels)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            #生成器损失

            g_loss=nn.BCELoss()(fake_score,one_labels)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_loss.step()

    print(f"epoch:{i+1},d_loss:{d_loss},g_loss:{g_loss}")

#如果是主文件
if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator=Generator().to(device)
    discriminator=Discriminator().to(device)
    g_optimizer=optim.Adam(generator.parameters())
    d_optimizer=optim.Adam(discriminator.parameters())
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset=datasets.CIFAR10('data',transform=transform,download=True)
    trainloder=DataLoader(dataset,batch_size=128,shuffle=True)
    train(generator,discriminator,10,trainloder,device,g_optimizer,d_optimizer)


