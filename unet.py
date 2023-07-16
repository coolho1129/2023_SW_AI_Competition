class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 64)

        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 128)

        self.dconv_down5 = double_conv(128, 256)
        self.dconv_down6 = double_conv(256, 256)

        self.dconv_down7 = double_conv(256, 512)    
        self.dconv_down8 = double_conv(512, 512) 

        self.dconv_down9 = double_conv(512, 1024)
        self.dconv_down10 = double_conv(1024, 1024)    

        self.maxpool = nn.MaxPool2d(2)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2, padding = 0)       

        self.dconv_up1 = double_conv(1024, 512)
        self.dconv_up2 = double_conv(512, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2, padding = 0)  

        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up4 = double_conv(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2, padding = 0)

        self.dconv_up5 = double_conv(256, 128)
        self.dconv_up6 = double_conv(128, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2, padding = 0    )

        self.dconv_up7 = double_conv(128, 64)
        self.dconv_up8 = double_conv(64, 64)

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

        x = self.upconv1(conv10)        
        x = torch.cat([x, conv8], dim=1)
 
        x = self.dconv_up1(x)
        x = self.dconv_up2(x)   
        x = self.upconv2(x)      
        x = torch.cat([x, conv6], dim=1)       

        x = self.dconv_up3(x)
        x = self.dconv_up4(x) 
        x = self.upconv3(x)       
        x = torch.cat([x, conv4], dim=1)   

        x = self.dconv_up5(x)
        x = self.dconv_up6(x)    
        x = self.upconv4(x)    
        x = torch.cat([x, conv2], dim=1)  

        x = self.dconv_up7(x)
        x = self.dconv_up8(x)  

        out = self.conv_last(x)

        return out

    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride = 1, bias = True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )   
