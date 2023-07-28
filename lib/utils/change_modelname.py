import torch
import torch.nn as nn
from lib.models.deepunet import * 
import os

MAINPATH=os.path.dirname(__file__)
os.chdir(MAINPATH)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = self.double_conv(3, 64)
        self.dconv_down2 = self.double_conv(64, 64)

        self.dconv_down3 = self.double_conv(64, 128)
        self.dconv_down4 = self.double_conv(128, 128)

        self.dconv_down5 = self.double_conv(128, 256)
        self.dconv_down6 = self.double_conv(256, 256)

        self.dconv_down7 = self.double_conv(256, 512)   
        self.dconv_down8 = self.double_conv(512, 512)

        self.dconv_down9 = self.double_conv(512, 1024)
        self.dconv_down10 = self.double_conv(1024, 1024)   

        self.dconv_down11 = self.double_conv(1024, 2048)
        self.dconv_down12 = self.double_conv(2048, 2048)  

        self.maxpool = nn.MaxPool2d(2)

        self.upconv1 = nn.ConvTranspose2d(2048, 1024, 2, 2, padding = 0)      

        self.dconv_up1 = self.double_conv(2048, 1024)
        self.dconv_up2 = self.double_conv(1024, 1024)


        self.upconv2 = nn.ConvTranspose2d(1024, 512, 2, 2, padding = 0)      

        self.dconv_up3 = self.double_conv(1024, 512)
        self.dconv_up4 = self.double_conv(512, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2, padding = 0) 

        self.dconv_up5 = self.double_conv(512, 256)
        self.dconv_up6 = self.double_conv(256, 256)

        self.upconv4 = nn.ConvTranspose2d(256, 128, 2, 2, padding = 0)

        self.dconv_up7 = self.double_conv(256, 128)
        self.dconv_up8 = self.double_conv(128, 128)

        self.upconv5 = nn.ConvTranspose2d(128, 64, 2, 2, padding = 0    )

        self.dconv_up9 = self.double_conv(128, 64)
        self.dconv_up10 = self.double_conv(64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(conv1)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        conv4 = self.dconv_down4(conv3)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        conv6 = self.dconv_down6(conv5)
        x = self.maxpool(conv6)

        conv7 = self.dconv_down7(x)
        conv8 = self.dconv_down8(conv7)
        x = self.maxpool(conv8)

        conv9 = self.dconv_down9(x)    
        conv10 = self.dconv_down10(conv9)
        x = self.maxpool(conv10)

        conv11 = self.dconv_down11(x)    
        conv12 = self.dconv_down12(conv11)

        x = self.upconv1(conv12)       
        x = torch.cat([x, conv10], dim=1)

        x = self.dconv_up1(x)
        x = self.dconv_up2(x)  
        x = self.upconv2(x)     
        x = torch.cat([x, conv8], dim=1)      

        x = self.dconv_up3(x)
        x = self.dconv_up4(x)
        x = self.upconv3(x)      
        x = torch.cat([x, conv6], dim=1)  

        x = self.dconv_up5(x)
        x = self.dconv_up6(x)   
        x = self.upconv4(x)   
        x = torch.cat([x, conv4], dim=1) 

        x = self.dconv_up7(x)
        x = self.dconv_up8(x)   
        x = self.upconv5(x)   
        x = torch.cat([x, conv2], dim=1) 

        x = self.dconv_up9(x)
        x = self.dconv_up10(x) 

        out = self.conv_last(x)

        return out

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride = 1, bias = True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )    

        
def save_model(model, path, epoch):
    if(epoch % 10 ==9):
        torch.save(model, path + str(epoch + 1) + ".pt")
        

def load_model(path):
    return torch.load(path)

def change_model_name(path):
    name=path.split('.')[1].split('/')[1] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=DeepUNet().to(device)
    #print(model)
    old_model=load_model(path)
    #print(old_model.state_dict())
    model.load_state_dict(old_model.state_dict())
    
    model_equal = True
    for model_state, old_state in zip(model.state_dict().values(), old_model.state_dict().values()):
        print(model_state,old_state)
        if not torch.equal(model_state, old_state):
            model_equal = False
            break

    # print(model_equal)
    if(model_equal):
        if(not os.path.exists('new_model') or os.path.isfile('new_model')):
            os.mkdir('new_model')
        torch.save(model, 'new_model/'+ name + ".pt")


  
    
#MODELPATH=""
#change_model_name(MODELPATH)   
    
    
  

