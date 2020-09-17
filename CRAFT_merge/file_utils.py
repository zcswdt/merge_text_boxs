# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import link_boxes_test
import link_y_boxes

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def find_rect_frame(rect):
    result_rect = []   
    x_min = min(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
    y_min = min(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
    x_max = max(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
    y_max = max(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
    result_rect.append([x_min, y_min, x_max, y_max])
    return result_rect


def saveResult(img_file, img, boxes, dirname='./result/',dirname_no = './result_no/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'
        res_img_file_no = dirname_no + "res_" + filename + '.jpg'
        
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        if not os.path.isdir(dirname_no):
            os.mkdir(dirname_no)

        if len(boxes) == 0:
            cv2.imwrite(res_img_file_no,img) 
        else:
            with open(res_file, 'w') as f:
                #保存角度校正以后的矩形框，坐标格式为左上角，和右下角
                rects= []
                for i, box in enumerate(boxes):
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    #print('1',poly)
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'
                    f.write(strResult)
    
                    poly = poly.reshape(-1, 2)
                    
                    #print('2',poly)
                    ##画多边形，顶点个数：4，矩阵变成4*1*2维
                    #print('3',poly.reshape((-1, 1, 2)))
                    #print('4',[poly.reshape((-1, 1, 2))].shape)
                    #print('5',[poly.reshape((-1, 1, 2))])
                    #cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                    
                    #求出文字框的外接矩形，(带有角度)
                    rect = find_rect_frame(poly)  #rect [[86, 158, 286, 180]]
                    rect = rect[0]    #[86, 158, 286, 180]
                    #保存矫正以后的文字框
                    
                    rects.append(rect)
                    
                    
                    #两个if语句是给文字识别用的接口，文字检测用不到。
                    ptColor = (0, 255, 255)
                    if verticals is not None:
                        if verticals[i]:
                            ptColor = (255, 0, 0)
    
                    if texts is not None:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                        cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
                print('rects',rects)
                #对文字框进行横向合并
                connector_x = link_boxes_test.X_BoxesConnector(rects, img.shape[1], max_dist=15, overlap_threshold=0.2)
                x_new_rects = connector_x.connect_boxes()
                print('x_new_rects',x_new_rects)
                #画出横向以后的文字框
                for rect in x_new_rects:
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)
                
                #在y轴方向对文字框进行合并
                connector_y = link_y_boxes.y_BoxesConnector(x_new_rects, img.shape[0], max_dist=15, overlap_threshold=0.001)
                y_new_rects = connector_y.connect_boxes()
                
                #画出y轴方向合并的文字框
                for rect in y_new_rects:
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
                
            # Save result image
            cv2.imwrite(res_img_file, img)
    
