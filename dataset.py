import metrics
import os
import numpy as np
import cv2
from torch.utils.data import Dataset


class BBoxRegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(BBoxRegressionDataset, self).__init__()
        self.transform = transform

        csv_path = os.path.join(root_dir, "bbox.csv")
        samples = np.loadtxt(csv_path, dtype=str)

        jpeg_list = list()
        # 保存{'image_id': ?, 'positive': ?, 'bndbox': ?}
        box_list = list()
        for i in range(len(samples)):
            sample_name = samples[i]

            jpeg_path = os.path.join(root_dir, "JPEGImages", sample_name + ".jpg")
            bndbox_path = os.path.join(root_dir, "bndboxs", sample_name + ".csv")
            positive_path = os.path.join(root_dir, "positive", sample_name + ".csv")

            jpeg_list.append(cv2.imread(jpeg_path))
            bndboxes = np.loadtxt(bndbox_path, dtype=int, delimiter=" ")
            positives = np.loadtxt(positive_path, dtype=int, delimiter=" ")

            if len(positives.shape) == 1:
                bndbox = self.get_bndbox(bndboxes, positives)
                box_list.append(
                    {"image_id": i, "positive": positives, "bndbox": bndbox}
                )
            else:
                for positive in positives:
                    bndbox = self.get_bndbox(bndboxes, positive)
                    box_list.append(
                        {"image_id": i, "positive": positive, "bndbox": bndbox}
                    )

        self.jpeg_list = jpeg_list
        self.box_list = box_list

    def __getitem__(self, index: int):
        assert index < self.__len__(), "数据集大小为%d，当前输入下标为%d" % (self.__len__(), index)

        box_dict = self.box_list[index]
        image_id = box_dict["image_id"]
        positive = box_dict["positive"]
        bndbox = box_dict["bndbox"]

        # 获取预测图像
        jpeg_img = self.jpeg_list[image_id]
        xmin, ymin, xmax, ymax = positive
        image = jpeg_img[ymin:ymax, xmin:xmax]

        if self.transform:
            image = self.transform(image)

        # 计算P/G的x/y/w/h
        target = dict()
        p_w = xmax - xmin
        p_h = ymax - ymin
        p_x = xmin + p_w / 2
        p_y = ymin + p_h / 2

        xmin, ymin, xmax, ymax = bndbox
        g_w = xmax - xmin
        g_h = ymax - ymin
        g_x = xmin + g_w / 2
        g_y = ymin + g_h / 2

        # 计算t
        t_x = (g_x - p_x) / p_w
        t_y = (g_y - p_y) / p_h
        t_w = np.log(g_w / p_w)
        t_h = np.log(g_h / p_h)

        return image, np.array((t_x, t_y, t_w, t_h))

    def __len__(self):
        return len(self.box_list)

    def get_bndbox(self, bndboxes, positive):
        """
        返回和positive的IoU最大的标注边界框
        :param bndboxes: 大小为[N, 4]或者[4]
        :param positive: 大小为[4]
        :return: [4]
        """

        if len(bndboxes.shape) == 1:
            # 只有一个标注边界框，直接返回即可
            return bndboxes
        else:
            scores = metrics.iou(positive, bndboxes)
            return bndboxes[np.argmax(scores)]


class CustomFinetuneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        img_path = os.path.join(root_dir, "JPEGImages")
        samples = [img_name.split(".jpg")[0] for img_name in os.listdir(img_path)]

        jpeg_images = [
            cv2.imread(os.path.join(root_dir, "JPEGImages", sample_name + ".jpg"))
            for sample_name in samples
        ]

        positive_annotations = [
            os.path.join(root_dir, "Annotations", sample_name + "_1.csv")
            for sample_name in samples
        ]
        negative_annotations = [
            os.path.join(root_dir, "Annotations", sample_name + "_0.csv")
            for sample_name in samples
        ]

        # 边界框大小
        positive_sizes = list()
        negative_sizes = list()
        # 边界框坐标
        positive_rects = list()
        negative_rects = list()

        for annotation_path in positive_annotations:
            rects = np.loadtxt(annotation_path, dtype=int, delimiter=" ")
            # 存在文件为空或者文件中仅有单行数据
            if len(rects.shape) == 1:
                # 是否为单行
                if rects.shape[0] == 4:
                    positive_rects.append(rects)
                    positive_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                positive_rects.extend(rects)
                positive_sizes.append(len(rects))
        for annotation_path in negative_annotations:
            rects = np.loadtxt(annotation_path, dtype=int, delimiter=" ")
            # 和正样本规则一样
            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    negative_rects.append(rects)
                    negative_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                negative_rects.extend(rects)
                negative_sizes.append(len(rects))

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_sizes = positive_sizes
        self.negative_sizes = negative_sizes
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects
        self.total_positive_num = int(np.sum(positive_sizes))
        self.total_negative_num = int(np.sum(negative_sizes))

    def __getitem__(self, index: int):
        # 定位下标所属图像
        image_id = len(self.jpeg_images) - 1
        if index < self.total_positive_num:
            # 正样本
            target = 1
            xmin, ymin, xmax, ymax = self.positive_rects[index]
            # 寻找所属图像
            for i in range(len(self.positive_sizes) - 1):
                if (
                    np.sum(self.positive_sizes[:i])
                    <= index
                    < np.sum(self.positive_sizes[: (i + 1)])
                ):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        else:
            # 负样本
            target = 0
            idx = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[idx]
            # 寻找所属图像
            for i in range(len(self.negative_sizes) - 1):
                if (
                    np.sum(self.negative_sizes[:i])
                    <= idx
                    < np.sum(self.negative_sizes[: (i + 1)])
                ):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

        # print('index: %d image_id: %d target: %d image.shape: %s [xmin, ymin, xmax, ymax]: [%d, %d, %d, %d]' %
        #       (index, image_id, target, str(image.shape), xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return self.total_positive_num + self.total_negative_num

    def get_positive_num(self) -> int:
        return self.total_positive_num

    def get_negative_num(self) -> int:
        return self.total_negative_num


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import random

    # bounding box test
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data_root_dir = "./data/bbox_regression"
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)

    print(len(data_set))
    rand_index = random.randint(0, len(data_set))
    print("index: ", rand_index)
    image, target = data_set[rand_index]
    print(image.shape)
    print(target)
    print(target.dtype)

    def test_fine_tune_dataset():
        from PIL import Image

        root_dir = "./data/fine_tune"
        train_data_set = CustomFinetuneDataset(root_dir)
        idx = random.randint(0, len(train_data_set))
        print("positive num: %d" % train_data_set.get_positive_num())
        print("negative num: %d" % train_data_set.get_negative_num())
        print("total num: %d" % train_data_set.__len__())

        # 测试id=3/66516/66517/530856
        image, target = train_data_set.__getitem__(idx)
        print("target: %d" % target)

        image = Image.fromarray(image)
        image.show()
        print(type(image))

    test_fine_tune_dataset()
