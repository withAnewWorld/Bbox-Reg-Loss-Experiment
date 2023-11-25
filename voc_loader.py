import random
import argparse
from PIL import ImageDraw
from torchvision.datasets import VOCDetection
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="where to download the Pascal VOC Detection dataset",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2007",
        choices=["2007", "2012"],
        help="year of Pascal VOC Detection dataset",
    )
    parser.add_argument(
        "--image-set",
        type=str,
        default="trainval",
        choices=["train", "trainval", "test"],
        help="Select the image_set to use(test is only available in 2007)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataset = VOCDetection(
        args.data_path, year=args.year, image_set=args.image_set, download=True
    )
    data_size = len(dataset)
    rand_idx = random.randint(0, data_size)
    img, target = dataset[rand_idx]
    bnd_boxes = utils.parse_xml(dataset.annotations[rand_idx])

    print("random index: ", rand_idx)
    print("bounding box coordinates:\n", bnd_boxes)
    print("image size: ", img.size)

    draw = ImageDraw.Draw(img)
    bnd_boxes = bnd_boxes.tolist()
    for bnd_box in bnd_boxes:
        draw.rectangle(bnd_box, outline=(0, 255, 0), width=3)

    img.show()
