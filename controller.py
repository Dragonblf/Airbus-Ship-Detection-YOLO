#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
"""

import argparse
import json
import os
import numpy as np
import sys
import tensorflow as tf
import keras.backend as K
from easydict import EasyDict as edict
from keras.backend import mean
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from pprint import pprint

import albumentations as augmentator
import cv2

from framework.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from framework.yolo3.utils import get_random_data


class YOLO(object):

    def __init__(self, mode, config_file):
        if mode not in ("Train", "Retrain", "Evaluate", "All"):
            print("Mode must be 'Train', 'Retrain', or 'Evaluate'")
            sys.exit(1)

        self.mode = mode
        self.config = config_file
        self.__init_all()

    def __init_all(self):
        print("==> Initializes variables.")
        self.__initialize_config()

        print("==> Initializes seeds.")
        self.__initialize_seeds()

        print("==> Initializes model.")
        self.model = self.__initialize_model()

        print("==> Initializes callbacks.")
        self.__initialize_callbacks()

        print("==> Initializes image augmentation pipeline.")
        self.augmentation_pipeline = self.__initialize_image_augmentation(self.input_shape)

        print("==> Create all needed directories.")
        self.__create_needed_dirs()

    def __initialize_config(self):
        try:
            if self.config is not None:
                with open(self.config, 'r') as f:
                    config_args_dict = json.load(f)
            else:
                print("Add a config file", file=sys.stderr)
                exit(1)
        except FileNotFoundError:
            print("ERROR: Config file not found: {}".format(self.config), file=sys.stderr)
            exit(1)
        except json.decoder.JSONDecodeError:
            print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
            exit(1)
        args = edict(config_args_dict)

        print("==> Loaded configurations:")
        #pprint(args)
        #print("\n")


        self.base_dir = args.experiment_dir
        self.anchors = np.array(
            [float(x) for x in args.anchors.split(',')]
        ).reshape(-1,2)
        self.labels = args.labels
        self.seed = args.seed
        self.annotation_file = os.path.join(self.base_dir, args.model.convert.train_annot_yolo_file)
        
        # Model data
        self.input_shape = (
            args.model.img_width,
            args.model.img_height,
            args.model.num_channels
        )
        self.show_summary = args.model.show_summary

        # Directories
        self.log_dir = os.path.join(self.base_dir, args.model.dir.log_dir)
        self.tensorboard_dir = os.path.join(self.log_dir, args.model.dir.tensorboard_dir)
        self.checkpoint_dir = os.path.join(self.log_dir, args.model.dir.checkpoint_dir)
        self.trained_dir = os.path.join(self.log_dir, args.model.dir.trained_dir)

        # Train and test      
        self.img_dir = args.model.train.train_image_folder 
        self.train_on_all_layers = args.model.train.train_on_all_layers
        self.num_epochs = args.model.train.num_epochs
        self.initial_epoch = args.model.train.initial_epoch
        self.batch_size = args.model.train.batch_size
        self.weights = args.model.train.weights
        self.verbose = args.model.train.verbose
        self.num_workers = args.model.train.num_workers
        self.use_multiprocessing = args.model.train.use_multiprocessing
        self.shuffle = args.model.train.shuffle
        self.ignore_tresh = args.model.train.ignore_tresh
        self.val_split = args.model.train.val_split

        if self.mode in ("Train", "Retrain"):
            self.num_images_to_train = args.model.train.num_images_to_train

        # TODO - Optimizer 
       
    def __initialize_seeds(self):
        np.random.seed(self.seed)

    def __initialize_model(self, freeze_body=2):
        K.clear_session()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        image_input = Input(shape=self.input_shape)
        num_anchors = len(self.anchors)
        num_classes = len(self.labels)
        w, h, _ = self.input_shape

        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                        num_anchors // 3, num_classes + 5)) for l in range(3)]
        model_body = yolo_body(image_input, num_anchors // 3, num_classes)
        print('==> Created YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if self.mode == "Retrain":
            model_body.load_weights(self.weights, by_name=True, skip_mismatch=True)
            print('==> Loaded weights from: {}'.format(self.weights))

            if freeze_body in (1,2):
                num = (185, len(model_body.layers) - 3)[freeze_body - 1]
                for i in range(num):
                    model_body.layers[i].trainable = False
                print('==> Freezed the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss,
                        output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': self.anchors,
                                   'num_classes': num_classes,
                                   'ignore_thresh': self.ignore_tresh,
                                   'print_loss': False}
                        )(model_body.output + y_true)
        model = Model(inputs=[model_body.input] + y_true, outputs=model_loss)

        if self.show_summary:
            model.summary()

        return model

    def __initialize_callbacks(self):
        self.callbacks = []
        self.callbacks += [TensorBoard(log_dir=self.tensorboard_dir)]
        self.callbacks += [ModelCheckpoint(
            os.path.join(self.checkpoint_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
            monitor='val_loss', save_weights_only=True,
            save_best_only=True, period=3
        )]
        self.callbacks += [ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=self.verbose
        )]
        self.callbacks += [EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10, verbose=self.verbose
        )]

    def __create_needed_dirs(self):
        for d in (self.log_dir, self.tensorboard_dir,
                  self.checkpoint_dir, self.trained_dir):
                  self._create_dir_if_not_exists(d)


    def train(self):
        print("==> Reading annotation file")
        with open(self.annotation_file, "r") as f:
            lines = f.readlines()

            if self.num_images_to_train:
                lines = lines[:self.num_images_to_train]
        print("==> Found {} lines".format(len(lines)))
        
        if self.shuffle:
            np.random.shuffle(lines)

        num_val = int(len(lines) * self.val_split)
        num_train = len(lines) - num_val

        if self.train_on_all_layers:
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable = True

            self.model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            print('==> Unfreeze all of the layers.')

        print('==> Train on {} samples, validate on {} samples, with batch size {}.'.format(num_train, num_val, self.batch_size))
        self.model.fit_generator(
            self._data_generator_wrapper(lines[:num_train]),
            steps_per_epoch = max(1, num_train // self.batch_size),
            validation_data = self._data_generator_wrapper(lines[num_train:]),
            validation_steps = max(1, num_val // self.batch_size),
            epochs = self.num_epochs,
            initial_epoch = self.initial_epoch,
            callbacks = self.callbacks,
            verbose = self.verbose
        )

        print("==> Saving final weights.")
        self.model.save_weights(os.path.join(self.trained_dir, 'trained_weights_final.h5'))

    def _data_generator_wrapper(self, annotation_lines):
        """
        For condition checking
        """
        n = len(annotation_lines)
        if n == 0 or self.batch_size <= 0: return None
        return self._data_generator(annotation_lines)

    def _data_generator(self, annotation_lines):
        '''
        Data generator for fit_generator
        '''
        n = len(annotation_lines)
        i = 0
        num_classes = len(self.labels)
        image_shape = (self.input_shape[0], self.input_shape[1])

        while True:
            image_data = []
            box_data = []

            for b in range(self.batch_size):
                if (i==0) and (self.shuffle):
                    np.random.shuffle(annotation_lines)

                # Get images and boxes
                #image, box = self._augmentate_image(annotation_lines[i])
                image, box = get_random_data(
                    annotation_lines[i], (self.input_shape[0], self.input_shape[1]),
                    random = True, img_folder = self.img_dir)
                # add pictures
                image_data.append(image)
                # Add box
                box_data.append(box)

                i = (i + 1) % n
            
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, image_shape, self.anchors, num_classes)    
            yield [image_data] + y_true, np.zeros(self.batch_size)

    def _augmentate_image(self, annotation_line):
        line = annotation_line.split()
        
        file = line[0].split(os.sep)[-1]
        image = cv2.imread(os.path.join(self.img_dir, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        ids = np.array([bboxes[i][4] for i in range(bboxes.shape[0])])
        bboxes = np.array([bboxes[i][:4] for i in range(bboxes.shape[0])])

        annotations = {
            "image": image,
            "bboxes": bboxes,
            "category_id": ids
        }

        annotated_data = self.augmentation_pipeline(**annotations)
        return annotated_data["image"], annotated_data["bboxes"]


    @staticmethod
    def __initialize_image_augmentation(image_shape):
        pipeline = augmentator.Compose(
            [
                augmentator.VerticalFlip(p=0.5),
                augmentator.HorizontalFlip(p=0.5),
                augmentator.RandomBrightness(limit=1.2, p=0.5),
                augmentator.RandomGamma(gamma_limit=37, p=0.5),
                augmentator.ElasticTransform(alpha=203, sigma=166, alpha_affine=106, p=0.5),
                augmentator.JpegCompression(quality_lower=25, quality_upper=100, p=0.5),
                augmentator.RandomContrast(limit=1, p=0.5),
                augmentator.Resize(image_shape[0], image_shape[1], p=1)
            ],
            bbox_params = {
                'format': 'coco',
                'label_fields': [ "category_id" ]
            }
        )
        return pipeline

    @staticmethod
    def _create_dir_if_not_exists(_dir):
        if not os.path.isdir(_dir):
            os.makedirs(_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLO - Controller")
    parser.add_argument("mode", type=str, metavar="MODE",
                        help="'Train', 'Retrain', or 'Evaluate'")
    parser.add_argument("-c", "--config", type=str, metavar="PATH",
                        help="Path to config file.")

    args = parser.parse_args()

    model = YOLO(args.mode, args.config)
    model.train()