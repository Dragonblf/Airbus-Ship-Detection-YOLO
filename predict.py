#!/usr/bin/env python
# -- coding: utf-8 --

"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from framework.yolo3.model import yolo_eval, yolo_body
from framework.yolo3.utils import letterbox_image
from framework.yolo3.utils import draw_boxes

class YOLO(object):
    def __init__(self, model_path, labels, anchors, image_size=None,
                 score_threshold=0.3, iou_threshold=0.5):
        """
        model_path (string):
            Direct path to model
        labels (list, tuple, np.array, ...):
            Labels to predict
        anchors (list of lists, list of tuples, ...):
            Anchors to use
        image_size (list/ tuple of two ints or None):
            Image size - Must be a multiple of 32
        score_threshold (float):
            Score threshold
        iou_threshold (float):
            IOU threshold
        """
        self.model_path = model_path

        self.score = score_threshold
        self.iou = iou_threshold
        self.class_names = labels
        self.anchors = np.array(anchors).reshape(-1,2)
        
        self.sess = K.get_session()
        self.model_image_size = image_size if image_size is not None else (None, None)

        self.colors = self.__get_colors(self.class_names)
        self.boxes, self.scores, self.classes = self.generate()

    @staticmethod
    def __get_colors(names):
        hsv_tuples = [(float(x) / len(names), 1., 1.) for x in range(len(names))] 
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)
        return colors

    def generate(self): 
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)  
        num_classes = len(self.class_names)  
        self.yolo_model = yolo_body(
            Input(
                shape=(self.model_image_size[0], self.model_image_size[1], 3)
            ),
            3,
            num_classes
        )
        self.yolo_model.load_weights(model_path)  

        print('model:{} , anchors:{} , and classes:{} are loaded.'.format(model_path, num_anchors, num_classes))

        # Filter box based on detection parameters
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output, self.anchors, len(self.class_names),
            self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
    
    
    def detect_video(self, video_path, output_path): 
        video_in = cv2.VideoCapture(video_path)
        width, height = int(video_in.get(3)), int(video_in.get(4))
        FPS = video_in.get(5)
        
        video_out = cv2.VideoWriter()
        video_out.open(output_path, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width, height))
        # video_out.open(output_path, int(video_in.get(cv2.CAP_PROP_FOURCC)), FPS, (width, height))
        
        width = np.array(width, dtype=float)
        height = np.array(height, dtype=float)
        image_shape = (height, width)
        
        while video_in.isOpened():
            ret, data = video_in.read()
            if ret==False: break
            video_array = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(video_array,mode='RGB')
            resized_image = image.resize(tuple(reversed(self.model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
                
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)   # Add batch dimension.
                
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={self.yolo_model.input: image_data,
                           self.input_image_shape: [image.size[1], image.size[0]],
                           K.learning_phase(): 0
                          })
            draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, self.colors)
            video_out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
                
        self.sess.close()
        video_in.release()
        video_out.release()
        print("Done.")
 
    def detect_image(self, image):
        start = timer()  

        if self.model_image_size != (None, None):  # 416x416, 416=32*13，Must be a multiple of 32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        
        print('detector size: {}'.format(image_data.shape))
        image_data /= 255.  
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  
        # thickness = (image.size[0] + image.size[1]) // 512  
        thickness = (image.size[0] + image.size[1]) // 300  
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]  
            box = out_boxes[i]  
            score = out_scores[i]  

            label = '{} {:.2f}'.format(predicted_class, score)  
            draw = ImageDraw.Draw(image)  
            label_size = draw.textsize(label, font)  

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom)) 

            if top - label_size[1] >= 0:  # Label text
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):  
                draw.rectangle([left + i, top + i, right - i, bottom - i],outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  
            del draw

        end = timer()
        print('time： {} s.'.format(end - start))
        return image

    def close_session(self):
        self.sess.close()
        
        
        
# def detect_img_for_test():
#     yolo = YOLO()
#     img_path = 'images/test.jpg'
#     image = Image.open(img_path)
#     r_image = yolo.detect_image(image)
#     yolo.close_session()
#     r_image.save('test001.png')


# if __name__ == '__main__':
#     detect_img_for_test()


