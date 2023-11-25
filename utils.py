import os
import random
import metrics
import xmltodict
import numpy as np
import matplotlib.pyplot as plt
import cv2
import selectivesearch
import torch
from torch.utils.data import Sampler


def parse_xml(xml_path):
    """
    解析xml文件，返回标注边界框坐标
    """

    with open(xml_path, "rb") as f:
        xml_dict = xmltodict.parse(f)

        bndboxs = list()
        objects = xml_dict["annotation"]["object"]
        if isinstance(objects, list):
            for obj in objects:
                bndbox = obj["bndbox"]
                bndboxs.append(
                    (
                        int(bndbox["xmin"]),
                        int(bndbox["ymin"]),
                        int(bndbox["xmax"]),
                        int(bndbox["ymax"]),
                    )
                )
        elif isinstance(objects, dict):
            bndbox = objects["bndbox"]
            bndboxs.append(
                (
                    int(bndbox["xmin"]),
                    int(bndbox["ymin"]),
                    int(bndbox["xmax"]),
                    int(bndbox["ymax"]),
                )
            )
        else:
            pass

        return np.array(bndboxs)


def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    """
    获取正负样本（注：忽略属性difficult为True的标注边界框）
    正样本：候选建议与标注边界框IoU大于等于0.5
    负样本：IoU大于0,小于0.5。为了进一步限制负样本数目，其大小必须大于标注框的1/5
    """
    img = cv2.imread(jpeg_path)

    selectivesearch.config(gs, img, strategy="q")
    # 计算候选建议
    rects = selectivesearch.get_rects(gs)
    # 获取标注边界框
    bndboxs = parse_xml(annotation_path)

    # 标注框大小
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # 获取候选建议和标注边界框的IoU
    iou_list = metrics.compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        if iou_list[i] >= 0.5:
            # 正样本
            positive_list.append(rects[i])
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            # 负样本
            negative_list.append(rects[i])
        else:
            pass

    return positive_list, negative_list


def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title("loss")
    plt.savefig("./loss.png")


def save_model(model, model_save_path):
    # 保存最好的模型参数
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)


class CustomBatchSampler(Sampler):
    def __init__(
        self, num_positive, num_negative, batch_positive, batch_negative
    ) -> None:
        """
        2分类数据集
        每次批量处理，其中batch_positive个正样本，batch_negative个负样本
        @param num_positive: 正样本数目
        @param num_negative: 负样本数目
        @param batch_positive: 单次正样本数
        @param batch_negative: 单次负样本数
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = list(range(length))

        self.batch = batch_negative + batch_positive
        self.num_iter = length // self.batch

    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate(
                (
                    random.sample(
                        self.idx_list[: self.num_positive], self.batch_positive
                    ),
                    random.sample(
                        self.idx_list[self.num_positive :], self.batch_negative
                    ),
                )
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter
