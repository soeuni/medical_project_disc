import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import get_df_stone, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC
from utils.util import *
from utils.torchsummary import summary
import shutil
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

Precautions_msg = '(주의사항) Stone dataset의 경우 사람당 4장의 이미지기때문에 batch사이즈를 4의 배수로 해야 제대로 평가 된다.'


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
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--use-ext', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')

    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--eval', type=str, choices=['best', 'best_no_ext', 'final'], default="best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='2,3,4')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--thresholds', type=float, default=0.5)  # PROBS에서 양성 판단 임계값 (acc 측정할때 사용)
    parser.add_argument('--select', type=int, default=300) # sorting된 confidence csv에서 가져올 이미지 개수


    args, _ = parser.parse_known_args()
    return args


def confidence_score(PROBS, TARGETS, PATIENT_ID, IMAGE_NAME, optimal_threshold):

    confidence_dict = {'image_name': list(np.concatenate(IMAGE_NAME)), 'patient_id': list(np.concatenate(PATIENT_ID)),
                       'prob': np.concatenate(PROBS)[:, 1],
                       'target': np.concatenate(TARGETS)
                       }

    # 정답을 맞춘 부분만 probs 표기, 이 외에는 모두 0
    # 1만 맞췄을때, 0과 1 모두 맞췄을 때 두가지 경우 존재 (한 level만 맞아도 정답으로 취급할땐, ==대신 or 연산자 사용)
    confidence_dict['correct_prob'] = (confidence_dict['prob'] >optimal_threshold) * confidence_dict['target'] * confidence_dict['prob']

    # dict로 dataframe 만들고 confidence_sum기준으로 sorting하여 저장
    confidence_csv = pd.DataFrame(confidence_dict)
    confidence_csv = confidence_csv.sort_values(by=['correct_prob'], axis=0, ascending=False)
    confidence_csv.to_csv(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_confidence.csv')


def sorted_confidence_save(select):

    # csv 만듬
    confidence_csv = pd.read_csv(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_confidence.csv')
    confidence_csv[:select].to_csv(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_high_confidence.csv', index=False)
    high_confidence = confidence_csv[['image_name']][:select].squeeze().tolist()

    for img_name in high_confidence: # 사진 저장
        shutil.copyfile(f'./data/images/train/{img_name}', f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_high_confidence_image/{img_name}')


def plot_roc_curve(fper, tper, fold, optimal_threshold):
    plt.figure()
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Optimal threshold: {optimal_threshold}')
    plt.legend()

    plt.savefig(f'./roc_curve_image/{args.kernel_type}_roc_curve_image/{args.kernel_type}_{fold}.png')
    # plt.show()


def val_epoch(model, loader, fold, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    PATIENT_ID = []
    IMAGE_NAME = []

    with torch.no_grad():
        for (data, target, patient_id, image_name) in tqdm(loader):

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
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            PATIENT_ID.append(patient_id)
            IMAGE_NAME.append(image_name)

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    PATIENT_ID = np.concatenate(PATIENT_ID)
    IMAGE_NAME = np.concatenate(IMAGE_NAME)

    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    # roc-curve
    fpr, tpr, threshold = roc_curve((TARGETS == 1).astype(float), PROBS[:, 1])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # roc plot
    plot_roc_curve(fpr, tpr, fold, optimal_threshold)

    # f1-score
    f1 = f1_score((TARGETS == 1).astype(float), PROBS[:, 1] > optimal_threshold)

    # recall-score
    recall = recall_score((TARGETS == 1).astype(float), PROBS[:, 1] > optimal_threshold)

    # precision
    precision = precision_score((TARGETS == 1).astype(float), PROBS[:, 1] > optimal_threshold)

    # auc
    auc = roc_auc_score((TARGETS == 1).astype(float), PROBS[:, 1])

    return LOGITS, PROBS, TARGETS, PATIENT_ID, IMAGE_NAME, val_loss, acc, f1, recall, precision, auc, optimal_threshold


def main():

    '''
    ####################################################
    # stone data 데이터셋 : dataset.get_df_stone
    ####################################################
    '''
    df_train, df_test, meta_features, n_meta_features  = get_df_stone(
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
    PATIENT_ID = []
    IMAGE_NAME = []


    folds = range(args.k_fold)

    for fold in folds:
        print(f'Evaluate data fold{str(fold)}')
        df_valid = df_train[df_train['fold'] == fold]

        # batch_normalization에서 배치 사이즈 1인 경우 에러 발생할 수 있으므로, 데이터 한개 버림
        if len(df_valid) % args.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid)-1)

        if args.DEBUG:
            df_valid = pd.concat([
                df_valid[df_valid['target'] == 1].sample(args.batch_size * 3),
                df_valid[df_valid['target'] != 1].sample(args.batch_size * 3)
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
        this_LOGITS, this_PROBS, this_TARGETS, this_PATIENT_ID, this_IMAGE_NAME, val_loss, acc, f1, recall, precision, auc, optimal_threshold = val_epoch(model, valid_loader, fold, is_ext=df_valid['is_ext'].values, n_test=8, get_output=True)
        LOGITS.append(this_LOGITS)
        PROBS.append(this_PROBS)
        TARGETS.append(this_TARGETS)
        PATIENT_ID.append(this_PATIENT_ID)
        IMAGE_NAME.append(this_IMAGE_NAME)

        # fold별 val_loss, acc, auc 출력
        print('validation loss : ', f'{val_loss:.6f}')
        print('Accuracy : ', f'{acc:.4f}')
        print('f1-score:', f'{f1:.4f}')
        print('recall-score:', f'{recall:.4f}')
        print('precision-score:', f'{precision:.4f}')
        print('auc: ', f'{auc:.4f}')
        print(classification_report(this_TARGETS, this_PROBS[:, 1] > optimal_threshold)) # fold 별 score 출력

    # 전체 fold 끝
    # 전체 데이터에 대한 confidence 구하고, confidence sum 기준으로 sorting
    confidence_score(PROBS, TARGETS, PATIENT_ID, IMAGE_NAME, optimal_threshold)

    # sorting 된 confidence_csv에서 상위 'select'개의 사진 저장
    sorted_confidence_save(args.select)

if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)
    os.makedirs(f'./confidence/{args.kernel_type}_confidence/{args.kernel_type}_high_confidence_image', exist_ok=True)
    os.makedirs(f'./roc_curve_image/{args.kernel_type}_roc_curve_image', exist_ok=True)

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
    criterion = nn.CrossEntropyLoss()

    main()

