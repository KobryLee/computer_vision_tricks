'''
  draw a picutre from the dataloader build by mmdet
  data=tensor()      shape is (3,H,W)
  
'''

from PIL import Image
import matplotlib.pyplot as plt
img=data['img'].data + torch.tensor([103.530, 116.280, 123.675]).reshape(-1,1,1)

# to (H,W,3)

img=torch.transpose(img,2,0)
img=torch.transpose(img,1,0)
img_np = np.uint8(img.numpy())
img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
image=Image.fromarray(img_np)
plt.imshow(image)
