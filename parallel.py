import os
import pickle
from multiprocessing import Pool
from collections import defaultdict
from datetime import datetime

def parallel_step(args):
    func, args_dict, pool_path = args
    result = {}
    for task_name, arg in args_dict.items():
        result[task_name] = func(arg)
    pickle.dump(result, open(pool_path, 'wb'))

def solve(message, sequential_cnt, parallel_cnt, func, args_dict, todo_list, output_dict_file):
    current_time = datetime.now()
    temp_path = output_dict_file.split('.')[0] + '_temp'
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    pool_path = os.path.join(temp_path, current_time.strftime("%Y%m%d_%H%M%S"))
    os.mkdir(pool_path)
    if not os.path.exists(output_dict_file):
        pickle.dump({}, open(output_dict_file, 'wb'))
    time_stamp = 0
    while True:
        time_stamp += 1
        output_dict = pickle.load(open(output_dict_file, 'rb'))
        unfinished_list = []
        for t in todo_list:
            if t not in output_dict:
                unfinished_list.append(t)
        if len(unfinished_list) == 0:
            print(f'{message}: finish at [time_stamp {time_stamp - 1}]!')
            return
        print(f'{message}: {len(unfinished_list)} tasks remained')
        div = defaultdict(list)
        for i, t in enumerate(unfinished_list):
            div[i // sequential_cnt].append(t)

        args_list = [(func, {task_name : args_dict[task_name] for task_name in task_list}, os.path.join(pool_path, f'{time_stamp}_{task_id}')) for task_id, task_list in div.items()]
        with Pool(parallel_cnt) as pool:
            pool.map(parallel_step, args_list)

        D = {}
        for file_name in os.listdir(pool_path):
            if file_name.startswith(f'{time_stamp}'):
                file_path = os.path.join(pool_path, file_name)
                D.update(pickle.load(open(file_path, 'rb')))
        output_dict.update(D)
        pickle.dump(output_dict, open(output_dict_file, 'wb'))

if __name__ == '__main__':
    pass