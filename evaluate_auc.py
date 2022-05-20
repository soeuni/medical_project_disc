import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import get_df_stone, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC
from utils.util import *
from utils.torchsummary import summary

Precautions_msg = '(주의사항) '


'''
- evaluate.py

학습한 모델을 평가하는 코드
Test셋이 아니라 학습때 살펴본 validation셋을 활용한다. 
grad-cam 실행한다. 


#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python evaluate.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> evaluate.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30 --k-fold 5



edited by MMCLab, 허종욱, 2020
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
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

    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--eval', type=str, choices=['best', 'best_no_ext', 'final'], default="best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--thresholds', type=float, default=0.5)  # PROBS에서 양성 판단 임계값 (acc 측정할때 사용)

    args, _ = parser.parse_known_args()
    return args


def val_epoch(model, loader, is_ext=None, n_test=1):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):

            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                # 5자리 target으로 변환
                change = np.zeros((len(target), args.out_dim))
                for i in range(len(target)):
                    change[i, :] = (list(list(format(target[i], '05'))))

                target = torch.tensor(change).float().to(device)
                data = data.to(device)

                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += m(l)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(m(logits), target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    # accuracy : 정확도
    acc = ((PROBS > args.thresholds) == TARGETS).mean() * 100

    # AUC : ROC 아래 넓이
    auc_list = np.zeros(args.out_dim)
    for level in range(args.out_dim):
        auc_list[level] = 0
        # auc_list[level] = roc_auc_score((TARGETS[:, level]).astype(float), PROBS[:, level])

    return LOGITS, PROBS, TARGETS, val_loss, acc, auc_list


def main():

    '''
    ####################################################
    # stone data 데이터셋 : dataset.get_df_stone
    ####################################################
    '''
    df_train, df_test, meta_features, n_meta_features = get_df_stone(
        k_fold = args.k_fold,
        out_dim = args.out_dim,
        data_dir = args.data_dir,
        data_folder = args.data_folder,
        use_meta = args.use_meta,
        use_ext = args.use_ext
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    LOGITS = []
    PROBS = []
    TARGETS = []


    folds = range(args.k_fold)
    for fold in folds:
        print(f'####Evaluate data fold{str(fold)}####')
        df_valid = df_train[df_train['fold'] == fold]

        # batch_normalization에서 배치 사이즈 1인 경우 에러 발생할 수 있으므로, 데이터 한개 버림
        if len(df_valid) % args.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid)-1)

        if args.DEBUG:
            df_valid = pd.concat([
                df_valid.sample(args.batch_size * 3),
                df_valid.sample(args.batch_size * 3)
            ])

        dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', meta_features, transform=transforms_val)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        elif args.eval == 'best_no_ext':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_no_ext_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

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

        '''
        ####################################################
        # stone data를 위한 평가함수 : val_epoch_stonedata
        ####################################################
        '''
        this_LOGITS, this_PROBS, this_TARGETS, val_loss, acc, auc_list = val_epoch(model, valid_loader, is_ext=df_valid['is_ext'].values, n_test=8)
        LOGITS.append(this_LOGITS)
        PROBS.append(this_PROBS)
        TARGETS.append(this_TARGETS)

        # fold별 val_loss, acc, auc 출력
        print('validation loss : ', f'{val_loss:.6f}')
        print('Accuracy : ', f'{acc:.4f}')
        print(f'L1-2 auc : {(auc_list[0]):.6f}, L2-3 auc : {(auc_list[1]):.6f}, L3-4 auc : {(auc_list[2]):.6f}, L4- auc : {(auc_list[3]):.6f}, L5-S1 auc : {(auc_list[4]):.6f}')



if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)

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
