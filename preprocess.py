import os
import utils
import selectivesearch
import numpy as np
import shutil
import metrics


if __name__ == "__main__":
    jpeg_dir = os.path.join(os.getcwd(), "data", "VOCdevkit", "VOC2007", "JPEGImages")

    dst_root_dir = "./data/bbox_regression/"
    dst_jpeg_dir = os.path.join(dst_root_dir, "JPEGImages")
    dst_bndbox_dir = os.path.join(dst_root_dir, "bndboxs")
    dst_positive_dir = os.path.join(dst_root_dir, "positive")

    os.makedirs(dst_root_dir, exist_ok=True)
    os.makedirs(dst_jpeg_dir, exist_ok=True)
    os.makedirs(dst_bndbox_dir, exist_ok=True)
    os.makedirs(dst_positive_dir, exist_ok=True)

    gt_annotation_dir = os.path.join(
        os.getcwd(), "data", "VOCdevkit", "VOC2007", "Annotations"
    )
    file_names = [
        xml_file_name.split(".xml")[0]
        for xml_file_name in os.listdir(gt_annotation_dir)
    ]
    res_samples = list()
    total_positive_num = 0
    for sample_name in file_names:
        jpeg_path = os.path.join(jpeg_dir, sample_name + ".jpg")

        gt_annotation_path = os.path.join(gt_annotation_dir, sample_name + ".xml")

        bndboxs = utils.parse_xml(gt_annotation_path)
        # 计算符合条件（IoU>0.6）的候选建议
        positive_list = list()

        gs = selectivesearch.get_selective_search()
        positive_bndboxes, negative_bncboxes = utils.parse_annotation_jpeg(
            gt_annotation_path, jpeg_path, gs
        )
        positive_bndboxes = np.array(positive_bndboxes)

        if len(positive_bndboxes.shape) == 1 and len(positive_bndboxes) != 0:
            scores = metrics.iou(positive_bndboxes, bndboxs)
            if np.max(scores) > 0.6:
                positive_list.append(positive_bndboxes)
        elif len(positive_bndboxes.shape) == 2:
            for positive_bndboxe in positive_bndboxes:
                scores = metrics.iou(positive_bndboxe, bndboxs)
                if np.max(scores) > 0.6:
                    positive_list.append(positive_bndboxe)
        else:
            pass

        # 如果存在正样本边界框（IoU>0.6），那么保存相应的图片以及标注边界框
        if len(positive_list) > 0:
            # 保存图片
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + ".jpg")
            shutil.copyfile(jpeg_path, dst_jpeg_path)
            # 保存标注边界框
            dst_bndbox_path = os.path.join(dst_bndbox_dir, sample_name + ".csv")
            np.savetxt(dst_bndbox_path, bndboxs, fmt="%s", delimiter=" ")
            # 保存正样本边界框
            dst_positive_path = os.path.join(dst_positive_dir, sample_name + ".csv")
            np.savetxt(
                dst_positive_path, np.array(positive_list), fmt="%s", delimiter=" "
            )

            total_positive_num += len(positive_list)
            res_samples.append(sample_name)
            print("save {} done".format(sample_name))
        else:
            print("-------- {} 不符合条件".format(sample_name))

    dst_csv_path = os.path.join(dst_root_dir, "bbox.csv")
    np.savetxt(dst_csv_path, res_samples, fmt="%s", delimiter=" ")
    print("total positive num: {}".format(total_positive_num))
    print("done")
