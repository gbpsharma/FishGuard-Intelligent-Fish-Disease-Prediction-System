import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
	def __init__(self,channels,reduction=16):
		super(ChannelAttention,self).__init__()
		self.fc1=nn.Linear(channels,channels//reduction)
		self.fc2=nn.Linear(channels//reduction,channels)
		self.relu=nn.ReLU()
		self.sigmoid=nn.Sigmoid()
	def forward(self,x):
		b,c,h,w=x.size()
		avg=torch.mean(x,dim=(2,3))
		att=self.fc1(avg)
		att=self.relu(att)
		att=self.fc2(att)
		att=self.sigmoid(att).view(b,c,1,1)
		return x*att

class MultiLayerResNet(nn.Module):
	def __init__(self):
		super(MultiLayerResNet,self).__init__()
		backbone=models.resnet50(pretrained=True)
		self.layer0=nn.Sequential(
			backbone.conv1,
			backbone.bn1,
			backbone.relu,
			backbone.maxpool,
			backbone.layer1
		)
		self.layer2=backbone.layer2
		self.layer3=backbone.layer3
		self.layer4=backbone.layer4
		self.attention=ChannelAttention(2048)
		self.pool=nn.AdaptiveAvgPool2d((1,1))
	def forward(self,x):
		x=self.layer0(x)
		f2=self.layer2(x)
		f3=self.layer3(f2)
		f4=self.layer4(f3)
		f4=self.attention(f4)
		f2=self.pool(f2).flatten(1)
		f3=self.pool(f3).flatten(1)
		f4=self.pool(f4).flatten(1)
		return torch.cat([f2,f3,f4],dim=1)

class OSELM(nn.Module):
	def __init__(self,input_dim,hidden_dim,output_dim):
		super(OSELM,self).__init__()
		self.hidden=nn.Linear(input_dim,hidden_dim)
		self.output=nn.Linear(hidden_dim,output_dim)
	def forward(self,x):
		return self.output(torch.relu(self.hidden(x)))

class FishDiseaseModel(nn.Module):
	def __init__(self,num_classes):
		super(FishDiseaseModel,self).__init__()
		self.backbone=MultiLayerResNet()
		self.classifier=OSELM(3584,1024,num_classes)
	def forward(self,x):
		return self.classifier(self.backbone(x))