import torch
import torch.nn as nn
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self,noise_len,h,w):
        super().__init__()
        self.h=h
        self.w=w
        self.fc=nn.Sequential(
            nn.Linear(noise_len,2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,self.h*self.w),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.fc(x)    #(b,noise_len)
        x=x.reshape(x.shape[0],1,self.h,self.w)
        return x
class Discriminator(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(h*w,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
        )
        def forward(self,x):
            x=x.reshape(x.shape[0],-1)
            x=self.fc(x)
            return x

noise_len=96
discriminator=Discriminator(28,28).to(device)
generator=Generator(noise_len,28,28).to(device)
d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=0.0002)
g_optimizer=torch.optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999)) #beta1 = 0.5：控制一阶矩（梯度的平均值）的衰减率。这个值较小有助于在训练开始时有更多的随机性，通常用于生成对抗网络（GAN）训练。beta2 = 0.999：控制二阶矩（梯度的平方的平均值）的衰减率。这个值较大，有助于稳定梯度。

for epoch in range(100):
    for i,(img,_) in enumerate(train_loader):
        real_img=img.to(device)
        batch=real_img.size(0)
        one_labels=torch.ones(batch,1).float().to(device) #1是为了与输出形状相同
        zero_labels=torch.zeros(batch,1).float().to(device)


        noise_data=torch.rand(batch,noise_len).to(device)#01均匀分布
        noise_data=(noise_data-0.5)/0.5 #(-1,1)对称利于训练稳定
        fake_img=generator(noise_data)
        fake_score=discriminator(fake_img)
        real_score=discriminator(real_img)

        d_loss=nn.BCEWithLogitsLoss()(real_score,one_labels)+nn.BCEWithLogitsLoss()(fake_score,one_labels)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        #再生成假图，打分，以此来指导生成器训练，所以判别器在生成器之前训练
        noise_data = torch.rand(batch, noise_len).to(device)  # 01均匀分布
        noise_data = (noise_data - 0.5) / 0.5  # (-1,1)对称利于训练稳定
        fake_img = generator(noise_data)
        fake_score = discriminator(fake_img)

        g_loss=nn.BCEWithLogitsLoss()(fake_score,one_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


