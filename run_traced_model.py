import torch
from PIL import Image
import numpy as np


model = torch.jit.load("style_vangogh_pretrained_cpu.pt")
img = Image.open("datasets/vangogh2photo/testB/2014-08-19 04:08:24.jpg")

img = np.array(img).transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0).to('cpu')
y = model(img)
out = y[0].detach().cpu().numpy().transpose(1, 2, 0)
img_out = Image.fromarray(out)
img_out.save("out.png")