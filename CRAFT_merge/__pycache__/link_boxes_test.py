import numpy as np


def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]


class X_BoxesConnector(object):
    def __init__(self, rects, imageW, max_dist=5, overlap_threshold=0.2):
        self.rects = np.array(rects)
        self.imageW = imageW
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageW)]  # 构建imageW个空列表
        for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[0]) < imageW:
                self.r_index[int(rect[0])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageW - 1].append(index)


    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            for idx in self.r_index[left]:
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:

                    return idx

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []       #相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any(): #优先级是not > and > or
                v = index

                sub_graphs.append([v])
       
                # 级联多个框(大于等于2个)

                while self.graph[v, :].any():

                    v = np.where(self.graph[v, :])[0][0]          #np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
             
                    sub_graphs[-1].append(v)

        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):
            proposal = self.get_proposal(idx)

            if proposal >= 0:

                self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1

        sub_graphs = self.sub_graphs_connected() #sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  #{0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])            #[[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:

            rect_set = self.rects[list(sub_graph)]     #[[228  78 238 128],[240  78 258 128]].....

            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


if __name__ == '__main__':
    import cv2
    rects = []
    rects.append(np.array([228, 78, 238, 128]))
    rects.append(np.array([240, 78, 258, 128]))
    rects.append(np.array([241, 130, 259, 140]))
    rects.append(np.array([79, 76, 127, 130]))
    rects.append(np.array([130, 76, 150, 130]))
    rects.append(np.array([152, 78, 172, 131]))

    rects.append(np.array([79, 150, 109, 180]))


    #创建一个白纸
    show_image = np.zeros([400, 400, 3], np.uint8) + 255


    connector = X_BoxesConnector(rects, 300, max_dist=5, overlap_threshold=0.3)
    new_rects = connector.connect_boxes()
    print(new_rects)

    for rect in rects:
        cv2.rectangle(show_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1)

    for rect in new_rects:
        cv2.rectangle(show_image,(rect[0], rect[1]), (rect[2], rect[3]),(255,0,0),1)
    cv2.imshow('res', show_image)
    cv2.waitKey(0)