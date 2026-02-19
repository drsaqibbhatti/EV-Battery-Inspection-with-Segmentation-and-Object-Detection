

import torch
import time


from torchsummary import summary
from model.SegNextV2 import SegNextV2


device = 'cuda' if torch.cuda.is_available() else 'cpu'



inputWidth = 640
inputHeight = 640
num_class = 1




model = SegNextV2(inputWidth=inputWidth,
                  inputHeight=inputHeight,
                  num_class=num_class).to(device)

summary(model, (3, 640, 640))


x = torch.randn((1,3,640,640)).to(device)
y = model.forward(x)



start_t = round(time.time() * 1000)
FPS = 0
while True:
    
    
    output = model(x)
    FPS = FPS + 1

    end_t = round(time.time() * 1000)

    if end_t - start_t > 1000:
        print('fps =', FPS)
        FPS = 0
        start_t = round(time.time() * 1000)