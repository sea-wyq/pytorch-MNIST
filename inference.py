import cv2
import torch
import torchvision.transforms as transforms
import time
from Model import MNIST


def images2tensor(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transf = transforms.ToTensor()
    img_tensor = torch.unsqueeze(transf(img), dim=0)
    return img_tensor


if __name__ == "__main__":
    start = time.time()
    device = torch.device('cpu')
    model = MNIST().to(device)
    model.load_state_dict(torch.load('mnist.pkl'))  # load
    input_data = images2tensor("0.png")
    res = model(input_data)
    end = time.time()
    print("手写数字图片检测的结果为：", res.argmax())
    print("infer time: ", end - start)
