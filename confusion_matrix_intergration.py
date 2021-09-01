import numpy as np
import csv
import os
import sys

def write_confusionmatrix_sum_csv(filepath):
    read_path = filepath + 'confusion_matrix/'
    #save_path = filepath + 'confusion_matrix/'

    file_list = sorted(os.listdir(read_path))
    confusion_matrix = np.zeros((22, 22), dtype=np.int32)
    for file_data in enumerate(file_list):
        i, file_name = file_data
        lists = []
        with open(read_path + file_name, 'r') as f:
            csv_file = csv.reader(f)
            for item in csv_file:
                lists.append(item)
        new_list = []
        lists.pop(0)
        for j in range(len(lists)):
            new_list.append(list(map(int, lists[j]))) # list str -> int
        list_np = np.array(new_list, dtype=np.int32)
        list_np = np.delete(list_np, 0, axis=1)
        print(i)
        confusion_matrix = confusion_matrix + list_np


    with open(read_path + '_confusion_matrix_sum.csv', 'w') as f:
        writer = csv.writer(f)
        confusion_matrix = confusion_matrix.tolist()
        for rows in enumerate(confusion_matrix):
            i, row = rows
            writer.writerow(row)

if __name__ == "__main__":
    write_confusionmatrix_sum_csv("/home/hagler/lettuce_segmentation_msrcnn/report/")