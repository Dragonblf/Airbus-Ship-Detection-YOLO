import argparse
import xml.etree.ElementTree as ET
import os
from glob import glob
from tqdm import tqdm


def convert_annotations(args):
    anno_dir = args.model.convert.anno_dir
    out_file = os.path.join(args.experiment_dir, args.model.convert.train_annot_yolo_file)
    labels = args.labels

    if not os.path.isdir(args.experiment_dir):
            os.makedirs(args.experiment_dir)

    with open(out_file, "w") as f:
        for xml in tqdm(glob(os.path.join(anno_dir, "*.xml"))):
            with open(xml, "r") as in_file:
                tree = ET.parse(in_file)
                root = tree.getroot()

                f.write(xml.replace(".xml", ".jpg"))
                
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in labels or int(difficult)==1:
                        continue

                    cls_id = labels.index(cls)
                    xmlbox = obj.find('bndbox')

                    x_min = int(xmlbox.find('xmin').text)
                    y_min = int(xmlbox.find('ymin').text)
                    x_max = int(xmlbox.find('xmax').text)
                    y_max = int(xmlbox.find('ymax').text)

                    f.write(" {},{},{},{},{}".format(x_min, y_min, x_max, y_max, cls_id))

                f.write("\n")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO - Convert annotations")
    parser.add_argument("-c", "--config", type=str, metavar="PATH",
                        help="Path to config file.")
    args = parser.parse_args()
    
    convert_annotations(args)