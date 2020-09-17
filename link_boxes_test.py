import numpy as np


def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]


class BoxesConnector(object):
    def __init__(self, rects, imageW, max_dist=None, overlap_threshold=None):
        print('max_dist',max_dist)
        print('overlap_threshold',overlap_threshold )
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
        print(self.r_index)

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        print('y1', y1)
        Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)

        print('Yaxis_overlap', Yaxis_overlap)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        print('rect',rect)

        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            #print('left',left)
            for idx in self.r_index[left]:
                print('58796402',idx)
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
                print('v',v)
                sub_graphs.append([v])
                print('sub_graphs', sub_graphs)
                # 级联多个框(大于等于2个)
                print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():

                    v = np.where(self.graph[v, :])[0][0]          #np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    print('v11',v)
                    sub_graphs[-1].append(v)
                    print('sub_graphs11', sub_graphs)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):

            proposal = self.get_proposal(idx)
            print('idx11', idx)
            print('proposal',proposal)
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
            print('1234', rect_set)
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


if __name__ == '__main__':
    import cv2
    rects = []
    # rects.append(np.array([228, 78, 238, 128]))
    # rects.append(np.array([240, 78, 258, 128]))
    # rects.append(np.array([241, 130, 259, 140]))
    # rects.append(np.array([79, 76, 127, 130]))
    # rects.append(np.array([130, 76, 150, 130]))
    # rects.append(np.array([152, 78, 172, 131]))
    #
    # rects.append(np.array([79, 150, 109, 180]))


    # #可以查到bug的一组
    # rects.append(np.array([97, 8, 465, 44]))
    # rects.append(np.array([6, 9, 80, 42]))
    # rects.append(np.array([3, 53, 178, 89]))
    # rects.append(np.array([184, 54, 280, 90]))
    #
    # rects.append(np.array([93, 145, 462, 185]))
    # rects.append(np.array([6, 149, 80, 184]))
    # rects.append(np.array([2, 197, 178, 230]))
    # rects.append(np.array([186, 198, 281, 229]))

    rects = [[144, 5, 192, 25], [25, 6, 64, 25], [66, 6, 141, 25], [193, 5, 275, 33], [269, 30, 354, 50], [26, 30, 182, 52],[185, 28, 265, 55], [25, 56, 89, 76], [93, 56, 229, 78], [232, 56, 262, 76], [264, 52, 343, 81]]




    #创建一个白纸
    show_image = np.zeros([500, 500, 3], np.uint8) + 255


    connector = BoxesConnector(rects, 500, max_dist=15, overlap_threshold=0.2)
    new_rects = connector.connect_boxes()
    print(new_rects)

    # for rect in rects:
    #     cv2.rectangle(show_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1)

    for rect in new_rects:
        cv2.rectangle(show_image,(rect[0], rect[1]), (rect[2], rect[3]),(255,0,0),1)
    cv2.imshow('res', show_image)
    cv2.waitKey(0)