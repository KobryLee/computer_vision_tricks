'''
  draw a picutre from the dataloader build by mmdet
  data=tensor()      shape is (3,H,W)
  
'''

from PIL import Image
import matplotlib.pyplot as plt

# when std =[1,1,1] or img need to be divided by std
img=weak['img'].data + torch.tensor([103.530, 116.280, 123.675]).reshape(-1,1,1)
# to (H,W,3)
img=torch.transpose(img,2,0)
img=torch.transpose(img,1,0)
img_np = np.uint8(img.numpy())
# tranfer BGR to RGB
img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
cv2.rectangle(img_np,(0,0),         #(x1,y1)
              (100,200),            #(x2,y2)
              (0,255,0),2)
image=Image.fromarray(img_np)      
plt.imshow(image)




# or we can add the mean after transpose to [h,w,3]
img = weak['img'].data
img=torch.transpose(img,2,0)
img=torch.transpose(img,1,0)
# img now is [h,w,3], 
img = img + torch.tensor([103.530, 116.280, 123.675])
# this is equal to  img[:,:,0] + mean[0].   img[:,:,1] + mean[1].  img[:,:,2] + mean[2]
