import torch 
import torch.nn.functional as F 

from dataset.transformation import to_tensor, inference_input
from model.lightning_wraper import HENet
from dataset.constant import ID2LABEL

from collections import OrderedDict

from PIL import Image
import cv2 as cv 
def load_model(checkpoint_path, n_classes): 
    model = HENet(n_classes=n_classes)
    checkpoint = torch.load(checkpoint_path)
    
    
    df_state_dict = OrderedDict()

    for k, v in checkpoint['state_dict'].items():
        name = k[6:] 
        df_state_dict[name]=v

    model.load_state_dict(df_state_dict)
    return model


if __name__ == "__main__":
    ck_path = "output/HENet-epoch=03-val_loss=0.0980-val_accuracy=0.9493.ckpt"
    img_path = "m4.png"
    cv_img = cv.imread(img_path)
    t_img = inference_input(cv_img)
    pil_img = Image.fromarray(cv.cvtColor(t_img, cv.COLOR_BGR2RGB)) 

    n_classes = 100

    model = load_model(checkpoint_path=ck_path, n_classes=n_classes)

    model.eval()
    ## Funtion begin 
    img_input = to_tensor(pil_img)
    img_input = torch.unsqueeze(img_input, 0)
    logit = model(img_input)
    probs = F.softmax(logit, dim=1)


    k = 5
    pred_score = torch.topk(probs, k=k).values.squeeze(dim=0).tolist()
    pred_id = torch.topk(probs, k=k).indices.squeeze(dim=0).tolist()
    ## Funtion end 

    print(f"Top {k} similar font:\n")
    for idx, pred in enumerate(pred_id): 
        print(f"{idx+1}. {ID2LABEL[pred]}, Score: {pred_score[idx]}")
