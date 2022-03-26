import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import MobileNetV2
from model_v3 import mobilenet_v3_small, mobilenet_v3_large


net_names = ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']
net_moduls = [MobileNetV2, mobilenet_v3_large, mobilenet_v3_small]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model and load model weights
    # model = MobileNetV2(num_classes=5).to(device)
    # model_weight_path = "mobilenet_v2.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    net_name = 'mobilenet_v3_small'
    module = dict(zip(net_names, net_moduls))[net_name]
    model = module(num_classes = 5).to(device)
    model_weight_path = f'{net_name}_update.pth' if os.path.isfile(f'./{net_name}_update.pth') else f'{net_name}.pth'
    pre_weights = torch.load(model_weight_path, map_location = device)

    pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict = False)
    
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
