import math
import cv2
import numpy as np
import csv

class Model_evaluator:
    def __init__(self):
        pass

    def get_center_point(self, bbox):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center_x = bbox[0] + (width / 2)
        center_y = bbox[1] + (height / 2)
        center_point = [center_x, center_y]
        return center_point

    def get_center_point_list(self, bbox_list):
        self.center_point_list = []
        for p in enumerate(bbox_list):
             i, bbox = p
             self.center_point_list.append(self.get_center_point(bbox))
        return self.center_point_list

    def get_point_dist(self, point1, point2):
        p1_x, p1_y = point1[0], point1[1]
        p2_x, p2_y = point2[0], point2[1]
        dist = math.sqrt(pow(p2_x-p1_x,2) + pow(p2_y-p1_y, 2))
        return dist

    def label_to_pred_matching(self, data):
        result_list = [[ [], [], [], [] ], [[], [], [], [], []]] # making result list
        for pred in enumerate(data[1][1]):
            i, pred_class = pred
            pred_center_point = self.get_center_point(data[1][2][i]) # get pred center point
            min_dist = float('inf')
            min_val = None
            for label in enumerate(data[0][1]):
                j, label_class = label
                label_center_point = self.get_center_point(data[0][2][j]) # get label center point
                dist = self.get_point_dist(pred_center_point, label_center_point) # get dist
                if min_dist > dist:
                    min_val = j
                    min_dist = dist
            result_list[0][0].append(data[0][0]) # label file name
            result_list[0][1].append(data[0][1][min_val]) # label class
            result_list[0][2].append(data[0][2][min_val])  # label bbox
            result_list[0][3].append(data[0][3][min_val])  # label poly
            result_list[1][0].append(data[1][0])  # pred file name
            result_list[1][1].append(data[1][1][i]) # pred class
            result_list[1][2].append(data[1][2][i])  # pred bbox
            result_list[1][3].append(data[1][3][i])  # pred poly
            result_list[1][4].append(data[1][4][i])  # pred value
        return result_list

    def swap(self, x, i, j):
        x[i], x[j] = x[j], x[i]

    def sorted_data(self, data):
        for size in reversed(range(len(data[0][1]))):
            max_i = 0
            for i in range(1, 1 + size):
                if data[0][1][i] > data[0][1][max_i]:
                    max_i = i
            data[0][1][max_i], data[0][1][size] = data[0][1][size], data[0][1][max_i]
            data[0][2][max_i], data[0][2][size] = data[0][2][size], data[0][2][max_i]
            data[0][3][max_i], data[0][3][size] = data[0][3][size], data[0][3][max_i]
            data[1][1][max_i], data[1][1][size] = data[1][1][size], data[1][1][max_i]
            data[1][2][max_i], data[1][2][size] = data[1][2][size], data[1][2][max_i]
            data[1][3][max_i], data[1][3][size] = data[1][3][size], data[1][3][max_i]
            data[1][4][max_i], data[1][4][size] = data[1][4][size], data[1][4][max_i]
        return data

    def get_polygon(self, top_predictions):
        polygon_list = []

        masks = top_predictions.get_field("mask").numpy()
        labels = top_predictions.get_field("labels")

        # img = np.where(masks[4] > 0, 255, 0)
        # img = np.squeeze(img, axis=0).astype(np.uint8)

        for data in enumerate(masks):
            i, mask = data
            thresh = mask[0, :, :, None]

            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            polygon_list.append(np.squeeze(contours[0], axis=1).tolist())
            #polygon_list.append(contours[0].tolist())

        return polygon_list

    def write_inference_csv(self, filepath, data):
        """
        data[0][0] = file_name
        data[0][1] = label.tolist()
        data[0][2] = label_bbox
        data[0][3] = targets[0].get_field('masks').polygons

        data[1][0] = file_name
        data[1][1] = pred_class
        data[1][2] = pred_bbox
        data[1][3] = polygon_list
        data[1][4] = top_predictions.get_field("scores").tolist()
        """

        with open(filepath, 'a') as f:
            writer = csv.writer(f)
            #writer.writerow(csv_list)
            for val in enumerate(data[0][0]):
                i, filename = val
                label = data[0][1][i]
                label_roi = data[0][2][i]
                label_poly = data[0][3][i].polygons[0].tolist()
                pred = data[1][1][i]
                pred_roi = data[1][2][i]
                pred_poly = data[1][3][i]
                pred_value = data[1][4][i]
                writer.writerow([filename, label, pred, pred_value, label_roi, pred_roi, label_poly, pred_poly])

    def write_confusionmatrix_csv(self, filepath, data):
        class_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        lists = []
        read_path = filepath + 'confusion_matrix_format.csv'
        save_path = filepath + 'confusion_matrix/'

        class_dict = {}
        for i in range(len(class_list)):
            class_dict[i] = class_list[i]

        confusion_matrix = np.zeros((22, 22), dtype=np.int8)
        for i in range(len(data[0][1])):
            #print(i)
            label_class = data[0][1][i]
            pred_class = data[1][1][i]
            label_idx = None
            pred_idx = None

            for key, value in class_dict.items():
                if label_class == value:
                    label_idx = key
            for key, value in class_dict.items():
                if pred_class == value:
                    pred_idx = key

            confusion_matrix[label_idx, pred_idx] += 1

        with open(read_path, 'r') as f:
            csv_file = csv.reader(f)
            for item in csv_file:
                lists.append(item)

        for i in range(len(lists)):
            if i == 0:
                continue
            for j in range(len(lists[i])):
                if j == 0:
                    continue
                lists[i][j] = confusion_matrix[i-1, j-1]


        with open(save_path + str(data[0][0][0]) +'_confusion_matrix.csv', 'w') as f:
            writer = csv.writer(f)
            for rows in enumerate(lists):
                i, row = rows
                writer.writerow(row)
