import xmltodict
import numpy as np


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
