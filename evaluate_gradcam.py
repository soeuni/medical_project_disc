import os
import argparse
import torch.nn as nn
from dataset import get_df_stone, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC
from utils.util import *

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

Precautions_msg = ' '

'''
- evaluate.py

학습한 모델을 평가하는 코드
Test셋이 아니라 학습때 살펴본 validation셋을 활용한다. 
grad-cam 실행한다. 


#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python evaluate.py --kernel-type 4fold_b3_300_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30

3d project 경우 밑 코드 실행
python evaluate.py --kernel-type 4fold_b3_10ep_alb01 --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --n-epochs 10 --image-size 300

pycharm의 경우: 
Run -> Edit Configuration -> evaluate.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b5_ns --n-epochs 30 --k-fold 4



edited by MMCLab, 허종욱, 2020

close
python evaluate.py --kernel-type 4fold_b3_15ep_reprint --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --sheave
fs
python evaluate.py --kernel-type 4fold_b3_30ep_fs --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --sheave

python evaluate.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b5_ns 


'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--image-size', type=int, default = 300)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dim', type=int, default=5)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--use-ext', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')

    parser.add_argument('--k-fold', type=int, default=4)
    parser.add_argument('--eval', type=str, choices=['best', 'best_no_ext', 'final'], default="best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str,  default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--use-gradcam')

    args, _ = parser.parse_known_args()
    return args


def main():
    '''
    ####################################################
    # stone data 데이터셋 : dataset.get_df_stone
    ####################################################
    '''
    df_train, df_test, meta_features, n_meta_features = get_df_stone(
        k_fold=args.k_fold,
        out_dim=args.out_dim,
        data_dir=args.data_dir,
        data_folder=args.data_folder,
        use_meta=args.use_meta,
        use_ext=args.use_ext
    )


    if args.eval == 'best':
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold0.pth')
    elif args.eval == 'best_no_ext':
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_no_ext_fold1.pth')
    if args.eval == 'final':
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold1.pth')

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim
    )
    model = model.to(device)

    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    ###############GRAD-CAM################
    target_layers = [model.enet.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # #grad cam에 넣을 사진 경로
    # path = './confidence/high_confidence_image/'
    high_confidence_csv = pd.read_csv(f'./confidence/{args.kernel_type}_high_confidence.csv')

    for idx in range(len(high_confidence_csv)):
        image = cv2.imread(f'./data/images/train/{high_confidence_csv["image_name"][idx]}')
        image = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image = cv2.resize(image, dsize=(456,456), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)
        image_float_np = image / 255
        input_tensor = torch.tensor(image).float()
        input_tensor = input_tensor.to(device).unsqueeze(0)

        grayscale_cam = cam(input_tensor=input_tensor)

        grayscale_cam = grayscale_cam[0, :]
        image_float_np = image_float_np.transpose(1, 2, 0)
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

        target = str(high_confidence_csv.loc[idx, 'target1':'target5'].tolist())
        prob = [round(high_confidence_csv.loc[idx, 'prob1':'prob5'].tolist()[j], 2) for j in range(5)]

        cv2.imwrite(f'./gradcam_image/{args.kernel_type}/{high_confidence_csv["confidence_mean"][idx]:.2f}_{target}_{prob}.png', cam_image)


        img = Image.open(f'./data/images/train/{high_confidence_csv["image_name"][idx]}')
        # draw = ImageDraw.Draw(img)
        #
        # draw.text((13, 13), f"{target} \n {prob}", font=ImageFont.truetype("C:\Windows\Fonts\gulim.ttc", 32), fill=(255, 255, 255))
        img.save(f'./gradcam_image/{args.kernel_type}/{high_confidence_csv["confidence_mean"][idx]:.2f}_{target}_{prob}.jpg')


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)
    os.makedirs(f'./gradcam_image/{args.kernel_type}', exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_MMC
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_MMC
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_MMC
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    criterion = nn.BCELoss()
    m = nn.Sigmoid()
    main()