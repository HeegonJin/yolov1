import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

def voc_parsing():
    name2class = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5, "horse": 6, "sheep": 7, "aeroplane": 8,
                  "bicycle": 9, "boat": 10, "bus": 11, "car": 12, "motorbike": 13, "train": 14, "bottle": 15,
                  "chair": 16,
                  "diningtable": 17, "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

    root_path = os.path.abspath(os.path.dirname(__file__))
    xml_path = os.path.join(root_path, "VOCdevkit", "VOC2007", "Annotations")
    img_path = os.path.join(root_path, "VOCdevkit", "VOC2007", "JPEGImages")

    annotation_list = glob.glob(os.path.join(xml_path, "*.xml"))
    label_list = []
    for annotation_file in annotation_list:
        _anno = {}
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        filename = root.find('filename').text
        image_file = os.path.join(img_path, filename)
        image_size = root.find("size")
        image_width = float(image_size.find("width").text)
        image_height = float(image_size.find("height").text)
        _objects = []
        for obj in root.iter("object"):
            obj_name = obj.find("name").text
            _class = float(name2class[obj_name])

            bndbox = obj.find("bndbox")
            _x_min = float(bndbox.find("xmin").text)
            _y_min = float(bndbox.find("ymin").text)
            _x_max = float(bndbox.find("xmax").text)
            _y_max = float(bndbox.find("ymax").text)
            _cx = (_x_min + _x_max) / 2 / image_width
            _cy = (_y_min + _y_max) / 2 / image_height
            _w_ratio = (_x_max - _x_min) / image_width
            _h_ratio = (_y_max - _y_min) / image_height
            _objects.append([_class, _cx, _cy, _w_ratio, _h_ratio])
        _anno[image_file] = _objects
        label_list.append(_anno)
    return label_list
