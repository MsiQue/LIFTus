import os
import time
import tqdm
import pandas as pd
from global_info import get_csv_folder
from sklearn.model_selection import train_test_split

def random_split_step(table_path, save_path, num, random_state):
    df = pd.read_csv(table_path, lineterminator='\n')
    if df.shape[0] < 2:
        return
    df1, df2 = train_test_split(df, test_size=0.5, random_state=random_state)
    table_name = table_path.split('/')[-1]
    save_path_1 = os.path.join(save_path, table_name.split('.')[0] + f'____{num}.csv')
    save_path_2 = os.path.join(save_path, table_name.split('.')[0] + f'____{num+1}.csv')
    df1.to_csv(save_path_1, index=False)
    df2.to_csv(save_path_2, index=False)

def random_split(dataset, times):
    save_path = f'positive_sampling/{dataset}'
    if os.path.exists(save_path):
        print(f'{dataset} already split')
        return
    os.makedirs(save_path)
    start_time = time.time()

    DATAFOLDER = get_csv_folder(dataset)

    for i in range(1, times + 1):
        j = i * 2 - 1
        solveList = []
        for table_name in tqdm.tqdm(os.listdir(DATAFOLDER)):
            table_path = os.path.join(DATAFOLDER, table_name)
            save_path_1 = os.path.join(save_path, table_name.split('.')[0] + f'____{j}.csv')
            if not os.path.exists(save_path_1):
                solveList.append(table_path)
        print(f'Tables remained : {len(solveList)}')
        for x in tqdm.tqdm(solveList):
            random_split_step(x, save_path, j, 42 + i)
    print(f'split {dataset} cost {time.time() - start_time}')

if __name__ == '__main__':
    random_split('test', 1)
    for d1 in ['SANTOS', 'TUS']:
        for d2 in ['small', 'large']:
            dataset = d1 + '_' + d2
            random_split(dataset, 1)
