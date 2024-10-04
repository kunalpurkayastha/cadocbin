import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vit_pytorch import ViT
from vit_pytorch.cross_vit import CrossViT
from models.binae import BinModel
from einops import rearrange
from torchvision import transforms
from PIL import Image
import torch
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


THRESHOLD = 0.5  
SPLITSIZE = 256  
setting = "base"  
patch_size = 16  
image_size = (SPLITSIZE, SPLITSIZE)


hyper_params = {
    "base": [6, 8, 768],
    "small": [3, 4, 512],
    "large": [12, 16, 1024]
}

encoder_layers = hyper_params[setting][0]
encoder_heads = hyper_params[setting][1]
encoder_dim = hyper_params[setting][2]


v = CrossViT(
    image_size=SPLITSIZE,
    num_classes=1000,
    depth=4,               
    sm_dim=192,            
    sm_patch_size=8,       
    sm_enc_depth=2,        
    sm_enc_heads=8,        
    sm_enc_mlp_dim=2048,   
    lg_dim=384,            
    lg_patch_size=16,      
    lg_enc_depth=3,        
    lg_enc_heads=8,        
    lg_enc_mlp_dim=2048,   
    cross_attn_depth=2,    
    cross_attn_heads=8,
    dropout=0.1,
    emb_dropout=0.1
)

model = BinModel(
    encoder=v,
    decoder_dim=encoder_dim,
    decoder_depth=encoder_layers,
    decoder_heads=encoder_heads
)

model_path = "./weights/best-model_8_2017base_256_8.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

deg_folder = './demo/natural/original/'
image_name = '5.jpg'
deg_image = cv2.imread(deg_folder + image_name) / 255

plt.imshow(deg_image[:, :, [2, 1, 0]])  
plt.title("Input Image")
plt.show()

def split(im, h, w):
    patches = []
    for ii in range(0, h, SPLITSIZE):
        for iii in range(0, w, SPLITSIZE):
            patches.append(im[ii:ii+SPLITSIZE, iii:iii+SPLITSIZE, :])
    return patches 

def merge_image(splitted_images, h, w):
    image = np.zeros((h, w, 3))
    ind = 0
    for ii in range(0, h, SPLITSIZE):
        for iii in range(0, w, SPLITSIZE):
            image[ii:ii+SPLITSIZE, iii:iii+SPLITSIZE, :] = splitted_images[ind]
            ind += 1
    return image  

h = ((deg_image.shape[0] // SPLITSIZE) + 1) * SPLITSIZE
w = ((deg_image.shape[1] // SPLITSIZE) + 1) * SPLITSIZE
deg_image_padded = np.ones((h, w, 3))
deg_image_padded[:deg_image.shape[0], :deg_image.shape[1], :] = deg_image
patches = split(deg_image_padded, deg_image.shape[0], deg_image.shape[1])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
out_patches = []
for p in patches:
    out_patch = np.zeros([3, *p.shape[:-1]])
    for i in range(3):
        out_patch[i] = (p[:, :, i] - mean[i]) / std[i]
    out_patches.append(out_patch)

result = []
for patch_idx, p in enumerate(out_patches):
    print(f"({patch_idx} / {len(out_patches) - 1}) processing patch...")
    p = np.array(p, dtype='float32')
    train_in = torch.from_numpy(p)

    with torch.no_grad():
        train_in = train_in.view(1, 3, SPLITSIZE, SPLITSIZE).to(device)
        _ = torch.rand((train_in.shape)).to(device)
        loss, _, pred_pixel_values = model(train_in, _)

        num_patches = pred_pixel_values.shape[1]
        patch_h = int(np.sqrt(num_patches))
        patch_w = patch_h
        p1, p2 = SPLITSIZE // patch_h, SPLITSIZE // patch_w

        rec_image = torch.squeeze(rearrange(pred_pixel_values, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p1, p2=p2, h=patch_h, w=patch_w))
        impred = rec_image.cpu().numpy()
        impred = np.transpose(impred, (1, 2, 0))
        for ch in range(3):
            impred[:, :, ch] = (impred[:, :, ch] * std[ch]) + mean[ch]
        impred[np.where(impred > 1)] = 1
        impred[np.where(impred < 0)] = 0
    result.append(impred)


clean_image = merge_image(result, deg_image_padded.shape[0], deg_image_padded.shape[1])
clean_image = clean_image[:deg_image.shape[0], :deg_image.shape[1], :]
clean_image = (clean_image > THRESHOLD) * 255


plt.imshow(clean_image.astype(np.uint8))
plt.title("Clean Image Output - Model") 
plt.show()
