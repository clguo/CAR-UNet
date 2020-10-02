import cv2
import os
# Please modify the path
path="Stare/Y/"
save="Stare/labels/"
for name in os.listdir(path):
    image = cv2.imread(path+name)
    cv2.imwrite(save+name.split(".")[0]+".png",image)

