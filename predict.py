import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import get_df_stone, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC
from utils.util import *
from sklearn.metrics import roc_auc_score

Precautions_msg = '(주의사항) Stone dataset의 경우 사람당 4장의 이미지기때문에 batch사이즈를 4의 배수로 해야 제대로 평가 된다.'

'''
- predict.py

학습한 모델을 이용해서, Test셋을 예측하는 코드
(Test셋에는 레이블이 존재하지 않음)

#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python predict.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> predict.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30

python predict.py --kernel-type 5fold_b7_100ep --data-folder original_stone/ --enet-type tf_efficientnet_b7_ns --n-epochs 100 --image-size 256
python predict.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30 --image-size 256
python predict.py --kernel-type 5fold_b3_256_30ep_ext --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30 --image-size 256 --use-ext

edited by MMCLab, 허종욱, 2020
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-ext', action='store_true')
    parser.add_argument('--k-fold', type=int, default=5)

    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_no_ext', 'final'], default="best")
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args


def main():

    '''
    ####################################################
    # stone data 데이터셋 : dataset.get_df_stone
    ####################################################
    '''
    df_train, df_test, meta_features, n_meta_features, target_idx = get_df_stone(
        k_fold = args.k_fold,
        out_dim = args.out_dim,
        data_dir = args.data_dir,
        data_folder = args.data_folder,
        use_meta = args.use_meta,
        use_ext = args.use_ext
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    # batch_normalization에서 배치 사이즈 1인 경우 에러 발생할 수 있음
    # 문제가 발생한 경우 배치 사이즈를 조정해서 해야한다.
    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)
    dataset_test = MMC_ClassificationDataset(df_test, 'test', meta_features, transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    PROBS = []
    folds = range(args.k_fold)
    for fold in folds:

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

        PROBS = []
        TARGETS = []
        with torch.no_grad():
            for (data, target) in tqdm(test_loader):

                if args.use_meta:
                    data, meta = data
                    data, meta = data.to(device), meta.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test):
                        l = model(get_trans(data, I), meta)
                        probs += l.softmax(1)
                else:   
                    data = data.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test):
                        l = model(get_trans(data, I))
                        probs += l.softmax(1)

                probs /= args.n_test

                '''
                ####################################################
                # 4장 의료 데이터를 묶음으로 확률계산. 평균이용
                # 타 프로젝트 진행시 삭제해야함
                ####################################################
                '''
                for b_i in range(int(data.shape[0] / 4)):
                    b_i4 = b_i * 4
                    probs[0 + b_i4:4 + b_i4, 0] = torch.mean(probs[0 + b_i4:4 + b_i4, 0])
                    probs[0 + b_i4:4 + b_i4, 1] = torch.mean(probs[0 + b_i4:4 + b_i4, 1])
                '''
                #################################################### 
                ####################################################            
                '''

                PROBS.append(probs.detach().cpu())
                TARGETS.append(target.detach().cpu())

        PROBS = torch.cat(PROBS).numpy()
        TARGETS = torch.cat(TARGETS).numpy()

        df_test['target'] = PROBS[:, target_idx]
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx])

        df_test[['image_name', 'target']].to_csv(os.path.join(args.sub_dir, f'sub_{args.kernel_type}_{args.eval}_{fold}_{acc:.2f}_{auc:.4f}.csv'), index=False)


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)
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

    main()
