import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

'''
image classification용 CSV 파일 만들때 주의할점
아래 두개는 반드시 포함해야한다. 

target: 클래스 번호. 예: {0, 1}
image_name: 이미지 파일 이름

'''

def get_df_stone(k_fold, data_dir, data_folder, out_dim = 1, use_meta = False, use_ext = False, side_task = 0):
    '''

    ##### get DataFrame
    데이터베이스 관리하는 CSV 파일을 읽어오고, 교차 validation을 위해 분할함
    stone 데이터셋을 위해 수정된 함수

    :param k_fold: argument에서 받은 k_fold 값
    :param out_dim: 네트워크 출력 개수
    :param data_dir: 데이터 베이스 폴더
    :param data_folder: 데이터 폴더
    :param use_meta: meta 데이터 사용 여부
    :param use_ext: 외부 추가 데이터 사용 여부

    :return:
    :target_idx 양성을 판단하는 인덱스 번호
    '''

    # # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    # df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    #
    # # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # # https://kongdols-room.tistory.com/123
    # # df_train = df_train[df_train['인덱스 이름'] != -1].reset_index(drop=True)
    #
    # # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    # # http://www.leejungmin.org/post/2018/04/21/pandas_apply_and_map/
    # df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x))  # f'{x}.jpg'
    #
    # # 원본데이터=0, 외부데이터=1
    # df_test['is_ext'] = 0

    # '''
    # ####################################################
    # 교차 검증 구현 (k-fold cross-validation)
    # ####################################################
    # '''
    # # image_len = len(df_train['patient_id'])
    # # patients_ids = len(df_train['patient_id'].unique())
    # # print(f'Original dataset의 사진 수 : {image_len}')
    # # print(f'Original dataset의 사람 인원수 : {patients_ids}')
    #
    # # 데이터 인덱스 : fold 번호. (fold)번 분할뭉치로 간다
    # # train.py input arg에서 k-fold를 수정해줘야함 (default:5)
    # print(f'Dataset: {k_fold}-fold cross-validation')
    #
    # # # 환자id : 분할 번호
    # # # 분할 방식을 나름대로 구현할 수 있다.
    # # filename2fold = {id: i % k_fold for i, id in enumerate(df_train['patient_id'].unique())}
    # # df_train['fold'] = df_train['patient_id'].map(filename2fold)
    #
    # df_train_new = []
    # for level in df_train['level'].unique():
    #     df = df_train[df_train['level'] == level].copy()
    #     filename2fold = {i: idx % 4 for idx, i in enumerate(df['image_name'])}
    #     df['fold'] = df['image_name'].map(filename2fold)
    #     df_train_new.append(df)
    # df_train = pd.concat(df_train_new)
    #
    # filename2fold = {}

    # '''
    # ####################################################
    # 외부 데이터를 사용할 경우에 대한 구현
    # ####################################################
    # '''
    # 외부 데이터를 사용할 경우 이곳을 구현
    if use_ext:
        # 외부 데이터베이스 경로
        ext_data_folder = 'ext_stone1/'

        # 외부 추가 데이터 (external data)
        # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
        df_train_ext = pd.read_csv(os.path.join(data_dir, ext_data_folder, 'train.csv'))

        df_train_ext['filepath'] = df_train_ext['image_name'].apply(
            lambda x: os.path.join(data_dir, f'{ext_data_folder}train', x))  # f'{x}.jpg'

        patients = len(df_train_ext['patient_id'].unique())
        print(f'External dataset의 사람 인원수 : {patients}')

        # 외부 데이터의 fold를 -1로 설정
        # fold에서 제외하면 validation에 사용되지 않고 항상 training set에 포함된다.
        df_train_ext['fold'] = -1

        # concat train data
        df_train_ext['is_ext'] = 1

        # # 데이터셋 전체를 다 쓰지 않고 일부만 사용
        # df_train_ext = df_train_ext.sample(1024)
        # df_train = pd.concat([df_train, df_train_ext]).reset_index(drop=True)



    # test data
    df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'



    '''
    ####################################################
    메타 데이터를 사용하는 경우 (나이, 성별 등)
    ####################################################
    '''
    # if use_meta:
    #     df_train, df_test, meta_features, n_meta_features = get_meta_data_stoneproject(df_train, df_test)
    # else:
    #     meta_features = None
    #     n_meta_features = 0
    meta_features = None
    n_meta_features = 0


    # '''
    # ####################################################
    # class mapping - 정답 레이블을 기록 (csv의 target)
    # ####################################################
    # 추간판 탈출증의 경우 csv에 target column 존재, 별도의 class mapping 필요 없음
    # '''
    # # target2idx = {d: idx for idx, d in enumerate(sorted(df_train.target.unique()))}
    # # df_train['target'] = df_train['target'].map(target2idx)
    # # target_idx = target2idx
    # if side_task == 1:
    #     diagnosis2idx2 = {d: idx for idx, d in enumerate(sorted(df_train.Age.unique()))}
    #     df_train['target2'] = df_train['Age'].map(diagnosis2idx2)

    # return df_train, df_test, meta_features, n_meta_features
    return df_test, meta_features, n_meta_features



class MMC_ClassificationDataset(Dataset):
    '''
    MMC_ClassificationDataset 클래스
    일반적인 이미지 classification을 위한 데이터셋 클래스
        class 내가만든_데이터셋(Dataset):
            def __init__(self, csv, mode, meta_features, transform=None):
                # 데이터셋 초기화

            def __len__(self):
                # 데이터셋 크기 리턴
                return self.csv.shape[0]

            def __getitem__(self, index):
                # 인덱스에 해당하는 이미지 리턴
    '''

    def __init__(self, csv, mode, meta_features, transform=None, side_task = 0):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode # train / valid
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform
        self.side_task = side_task

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 이미지 tranform 적용
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        # 메타 데이터를 쓰는 경우엔 image와 함께 텐서 생성
        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        # if self.mode == 'test':
        #     # Test 의 경우 정답을 모르기에 데이터만 리턴
        #     return data
        # else:
        #     # training 의 경우 CSV의 스톤여부를 타겟으로 보내줌
        if self.mode == 'valid':
            # return data, torch.tensor(self.csv.iloc[index].target).long(), self.csv.iloc[index].patient_id,self.csv.iloc[index].image_name
            return data, torch.tensor(self.csv.iloc[index].target).long(), self.csv.iloc[index].image_name

        else:
            if self.side_task:
                return data, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(self.csv.iloc[index].target2).long()
            else:
                return data, torch.tensor(self.csv.iloc[index].target).long()




def get_transforms(image_size):
    '''
    albumentations 라이브러리 사용함
    https://github.com/albumentations-team/albumentations

    TODO: FAST AUTO AUGMENT
    https://github.com/kakaobrain/fast-autoaugment
    DATASET의 AUGMENT POLICY를 탐색해주는 알고리즘

    TODO: Unsupervised Data Augmentation for Consistency Training
    https://github.com/google-research/uda

    TODO: Cutmix vs Mixup vs Gridmask vs Cutout
    https://www.kaggle.com/saife245/cutmix-vs-mixup-vs-gridmask-vs-cutout

    '''
    # 서버컴이랑 비교


    transforms_test = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_test
