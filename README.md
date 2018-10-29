# Airbus-Ship-Detection-YOLO

YOLOv3 framework with some helper utilities to train a model for the competition [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection). It's one of some parts of the hole network.

# TODO
- [ ] Test it
- [ ] Initializer.sh labels as list
- [ ] Add examples
- [ ] Add command-line support
- [ ] Update README

## Usage
0. Clone this repo to your local system.
    ```
    git clone https://github.com/Dragonblf/Airbus-Ship-Detection-YOLO
    ```    
1. Firstly you have to install the needed libraries. You can do it by yourself or just run 'installer.sh'. You have to install graphiv into your environment variables.
    ```
    sh installer.sh
    ```
    ```
    python -r install requirements.txt
    apt-get install graphviz
    ```
2. Secondly you have to convert your annotation (Pascal-VOC format), download the official YOLOv3 weights and ccnvert config and weights files to Keras h5 model.
    ```
    sh initializer.sh ANNOTATION_PATH LABEL OUTPUT_PATH
    ```
    ```
    # 1. Convert VOC-Annoations to the needed YoloV3 format
    python framework/convert/convert_annotations.py --annotations=PATH --labels=LABEL, --output=PATH
    
    # 2. Download official yolov3.weigths
    wget https://pjreddie.com/media/files/yolov3.weights -P framework/darknet/weights/yolov3.weights
    
    # 3. Convert config and weights to Keras h5 model
    python framework/convert/convert.py framework/darknet/configs/yolov3.cfg framework/darknet/weights/yolov3.weights framework/model_data/yolo3.h5
    rm framework/darknet/weights/yolov3.weights
    ```
    # --- UPDATE --- #
    ```
    ##### For some reasons the annotation convertion doesn't work on kaggle trough the function, but you can execute the function itself
    ##### in your cell. Just copy and paste it.
    ##### out_file: path to the output file - should be a .txt
    ##### anno_dir: path to the directory containing all annotations in Pascal-VOC format.
    ##### labels: list of all used labels for the training.
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
    ```
3. Now you have everything to train the model. To train your model, you have to make a YOLO class from 'train.py'.
    ```python
    from train import YOLO
    
    yolo = YOLO(annotation_path, labels, anchors, 
                 image_size, num_epochs, weights, log_dir
                 batch_size, initial_epoch, finetuning,
                 optimizer, pretrained_model, ignore_tres,
                 tensorboard_logging, model_checkpoint,
                 reduce_lr_on_plateau, early_stopping,
                 seed)
    ```
    Parameters are:
    - annotation_path (string):
        Path to the annotation file
    - labels (list, tuple, np.array, ...):
        Labels to predict
    - anchors (list of lists, list of tuples, ...):
        Anchors to use
    - image_size (list/ tuple of two ints):
        Image size - Must be a multiple of 32
    - num_epochs (int):
        Number of epochs to train
    - weights (string)
        Path to the weights file
    - log_dir (string):
        Path to the logging directory
    - batch_size (int):
        Batch size used in training
    - initial_epoch (int):
        Epoch at which to start training (useful for resuming a previous training run)
    - finetuning (True or False):
        True to finetune the model
    - optimizer (Keras optimizer):
        Optimizer for training
    - pretrained_model (string or None):
        Path to the pretrained model
    - ignore_tresh (float):
        Value of tresh ignore (used in layer yolo_loss)
    - tensorboard_logging (Keras callback or None):
        Callback for training. If None, standard will be used
    - model_checkpoint (Keras callback or None):
        Callback for training. If None, standard will be used
    - reduce_lr_on_plateau (Keras callback or None):
        Callback for training. If None, standard will be used
    - early_stopping (Keras callback or None):
        Callback for training. If None, standard will be used
    - seed (int):
        Random seed