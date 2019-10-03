# coding:utf-8

import numpy as np
from joblib import load



from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider

import cv2
import tensorflow as tf

from sheet_tracking import SheetTracker
from prediction import process, detect_corners, transform_sheet, order_points
from centroid_tracking import CentroidTracker
from kivy_design import MyScatterLayout, FloatInput, IntInput
from kivy.config import Config
from camera_handler import Camera

# Handle Camera


Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75),
                                  device_count={'GPU': 1})
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

tf.keras.backend.set_session(sess)

detector = CentroidTracker(20, 50)
# cap = cv2.VideoCapture(2)
model_svc = load('model1.joblib')
model_dnn = tf.keras.models.load_model('test2-4-conv-2-dense_Final')
# model_dnn = tf.keras.models.load_model('epic_num_reader.model')
path = './images3/'
count = 0


class ConfigurationLayout(FloatLayout):
    # runs on initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 1

        self.total_document = 0
        self.sheet_detection = False
        self.rotation_value = 0
        self.total_page = 0
        self.threshold_detection = False
        self.threshold_value = 85
        self.smooth_values = 1


        # Rotation
        self.label_rotate = Label(text="Rotation", pos_hint={"x": .05, "y": .9}, size_hint=(.9, .1), font_size='20sp')
        self.add_widget(self.label_rotate)

        self.btn_rotate_clockwise = Button(text="Droite", pos_hint={"x": .5, "y": .87}, size_hint=(.45, .05))
        self.btn_rotate_clockwise.bind(on_press=self.rotate_clockwise)
        self.add_widget(self.btn_rotate_clockwise)

        self.btn_rotate_counter_clockwise = Button(text="Gauche", pos_hint={"x": .05, "y": .87}, size_hint=(.45, .05))
        self.btn_rotate_counter_clockwise.bind(on_press=self.rotate_counter_clockwise)

        self.add_widget(self.btn_rotate_counter_clockwise)

        # Sheet Detection
        self.label_sheet_detection = Label(text="Détection de feuille", pos_hint={"x": .05, "y": .77},
                                           size_hint=(.9, .1), font_size='20sp', halign="left", valign="middle")
        self.label_sheet_detection.bind(size=self.label_sheet_detection.setter('text_size'))
        self.add_widget(self.label_sheet_detection)

        self.checkbox_sheet_detection = CheckBox(pos_hint={"x": .9, "y": .77}, size_hint=(.05, .1))
        self.add_widget(self.checkbox_sheet_detection)
        self.checkbox_sheet_detection.bind(active=self.on_checkbox_active)

        # threshold
        self.threshold_active_label = Label(text="Zone de détection", pos_hint={"x": .05, "y": .70},
                                           size_hint=(.9, .1), font_size='20sp', halign="left", valign="middle")
        self.threshold_active_label.bind(size=self.threshold_active_label.setter('text_size'))
        self.add_widget(self.threshold_active_label)

        self.checkbox_threshold = CheckBox(pos_hint={"x": .9, "y": .70}, size_hint=(.05, .1))
        self.add_widget(self.checkbox_threshold)
        self.checkbox_threshold.bind(active=self.checkbox_threshold_active)

        self.threshold_slider = Slider(min=0, max=100, value=self.threshold_value,  pos_hint={"x": .05, "y": .65},
                                       size_hint=(.9, .1))
        self.threshold_slider.disabled = True
        self.threshold_slider.bind(value=self.slider_threshold_changed)
        self.add_widget(self.threshold_slider)

        # Smooth values
        self.label_smooth_values = Label(text="Lisser les données", pos_hint={"x": .05, "y": .55}, size_hint=(.9, .1),
                                         font_size='20sp')
        self.add_widget(self.label_smooth_values)

        self.smooth_values_slider = Slider(min=1, max=50, value=self.smooth_values, pos_hint={"x": .05, "y": .50},
                                       size_hint=(.9, .1))
        self.smooth_values_slider.bind(value=self.slider_smooth_values_changed)
        self.add_widget(self.smooth_values_slider)

        # change camera
        self.label_camera_change = Label(text="Caméra numéro ", pos_hint={"x": .05, "y": .35},
                                           size_hint=(.9, .1), font_size='20sp', halign="left", valign="middle")
        self.label_camera_change.bind(size=self.label_camera_change.setter('text_size'))
        self.add_widget(self.label_camera_change)

        self.input_camera = IntInput(font_size='20sp', pos_hint={"x": .50, "y": .37}, size_hint=(.1, .05))
        self.add_widget(self.input_camera)

        # Set total to 0
        self.btn_set_to_zero = Button(text="Mettre points à 0", pos_hint={"x": .05, "y": .3}, size_hint=(.45, .05))
        self.btn_set_to_zero.bind(on_press=self.set_to_zero)
        self.add_widget(self.btn_set_to_zero)

        # add selected points
        self.btn_add_point = Button(text="Ajouter :", pos_hint={"x": .5, "y": .3}, size_hint=(.3, .05))
        self.btn_add_point.bind(on_press=self.add_point)
        self.add_widget(self.btn_add_point)

        self.input_points = FloatInput(font_size='20sp', pos_hint={"x": .80, "y": .3}, size_hint=(.15, .05))
        self.add_widget(self.input_points)
        # Total
        self.label_total_page = Label(text="Total pour cette page : 0", pos_hint={"x": .05, "y": .2},
                                      size_hint=(.9, .1), font_size='20sp', halign="left", valign="middle")
        self.label_total_page.bind(size=self.label_total_page.setter('text_size'))
        self.add_widget(self.label_total_page)

        self.label_total_document = Label(text="Total pour ce document : 0", pos_hint={"x": .05, "y": .15},
                                          size_hint=(.9, .1), font_size='20sp', halign="left", valign="middle")
        self.label_total_document.bind(size=self.label_total_document.setter('text_size'))
        self.add_widget(self.label_total_document)

        self.btn_add_total = Button(text="Ajouter total!", pos_hint={"x": .05, "y": .05},
                                    size_hint=(.9, .1))  # background_color=(.76, .89, .93, 1)
        self.btn_add_total.bind(on_press=self.add_total)
        self.add_widget(self.btn_add_total)

    def update_total_page(self, total):
        self.total_page = total
        self.label_total_page.text = "Total pour cette page : {}".format(self.total_page)

    def set_to_zero(self, instance):
        self.total_document = 0
        self.label_total_document.text = "Total pour ce document : {}".format(self.total_document)

    def add_point(self, instance):
        try:
            self.total_document += float(self.input_points.text)
            self.total_document = round(self.total_document, 2)
        except:
            pass

        self.label_total_document.text = "Total pour ce document : {}".format(self.total_document)


    def add_total(self, instance):
        self.total_document += self.total_page
        self.label_total_document.text = "Total pour ce document : {}".format(self.total_document)

    def on_checkbox_active(self, checkbox, value):
        self.sheet_detection = value

    def checkbox_threshold_active(self, checkbox, value):
        self.threshold_detection = value
        if value:
            self.threshold_slider.disabled = False
        else:
            self.threshold_slider.disabled = True

    def slider_threshold_changed(self, instance, value):
        self.threshold_value = value

    def slider_smooth_values_changed(self, instance, value):
        self.smooth_values = value
        # print(value)

    def rotate_clockwise(self, instance):
        self.rotation_value -= 90

    def rotate_counter_clockwise(self, instance):
        self.rotation_value += 90


class CorrectPage(FloatLayout):
    # runs on initialization
    def __init__(self, capture, sheet_tracker, **kwargs):
        super().__init__(**kwargs)
        self.capture = capture
        self.sheet_tracker = sheet_tracker

        self.cols = 2  # used for our grid

        self.my_camera = KivyCamera(capture=self.capture, sheet_tracker=self.sheet_tracker, fps=30, size_hint=(.7, 1),
                                    allow_stretch=True)
        self.add_widget(self.my_camera)

        self.configuration_layout = ConfigurationLayout(pos_hint={"x": .7}, size_hint=(.3, 1))
        self.add_widget(self.configuration_layout)

        self.mini_camera_layout = MyScatterLayout(x_by_y=1.3, do_rotation=False, size=(300, 200),
                                                  size_hint=(None, None), pos=(10, 10))
        self.mini_camera = MiniKivyCamera()
        self.mini_camera_layout.add_widget(self.mini_camera)
        self.add_widget(self.mini_camera_layout)


class MiniKivyCamera(Image):
    def __init__(self, **kwargs):
        super(MiniKivyCamera, self).__init__(**kwargs)

    def update(self, frame):
        # convert it to texture
        cv2.resize(frame, tuple(map(int, self.parent.size)))
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture

    def on_touch_down(self, touch):
        return False


class KivyCamera(Image):
    def __init__(self, capture, sheet_tracker, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        self.sheet_tracker = sheet_tracker
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.available_cameras = self.capture.list_video_devices_linux()
        self.capture.switch_camera(self.available_cameras[0])


    def update(self, dt):

        previous = self.capture.idx
        if self.capture.idx is None or self.parent.configuration_layout.input_camera.text == "":
            self.parent.configuration_layout.input_camera.text = str(self.available_cameras[0])
        if not int(self.parent.configuration_layout.input_camera.text) == self.capture.idx:
            if self.capture.switch_camera(int(self.parent.configuration_layout.input_camera.text)):
                self.parent.configuration_layout.input_camera.text = str(self.capture.idx)
            else:
                self.parent.configuration_layout.input_camera.text = str(previous)
        ret, frame = self.capture.video_capture.read()
        # ret = True
        # frame = np.uint8(np.ones((480, 640, 3)) * 200)
        # cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 2)
        if ret:
            showed_img = frame.copy()

            #get all the usefull values from the configuration layout class
            sheet_detection = self.parent.configuration_layout.sheet_detection
            rotation_value = self.parent.configuration_layout.rotation_value
            threshold_detection = self.parent.configuration_layout.threshold_detection
            threshold_value = self.parent.configuration_layout.threshold_value
            smooth_values = self.parent.configuration_layout.smooth_values



            #apply the rotation to the frame
            rows, cols = frame.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_value, 1)
            frame = cv2.warpAffine(frame, M, (cols, rows))

            # show the image in the mini camera view
            self.parent.mini_camera.update(frame)

            # detect the sheet
            if sheet_detection:
                approx = detect_corners(frame)

                if approx is not False:
                    approx = order_points(approx)
                    self.sheet_tracker.nbrs_in_mean = smooth_values
                    self.sheet_tracker.update(approx)
                    mean = self.sheet_tracker.get_mean()
                    new_approx = np.ndarray((4, 2), dtype='float32')
                    for c in range(len(approx)):
                        dist = (((approx[c][0]-mean[c][0])**2)+((approx[c][1]-mean[c][1])**2))**(1/2)
                        if dist < 40:
                            new_approx[c] = np.float32(approx[c])
                        else:
                            new_approx[c] = np.float32(mean[c])
                    frame = transform_sheet(frame, new_approx)
                    if frame is False:
                        return
                else:
                    return

            # do the detection
            result = process(frame, model_svc, model_dnn, detector, threshold_detection, threshold_value)
            if result is False:
                return
            (self.total_page, showed_img) = result

            # resize the image to the entire layout
            # showed_img = cv2.resize(showed_img, tuple(map(int, self.size)))

            # convert it to texture
            buf1 = cv2.flip(showed_img, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(showed_img.shape[1], showed_img.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
            # cv2.wasitKey(1)
            self.parent.configuration_layout.update_total_page(self.total_page)


class CamApp(App):
    def build(self):
        self.capture = Camera()
        self.sheet_tracker = SheetTracker()
        # self.my_camera
        return CorrectPage(self.capture, self.sheet_tracker)

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.video_capture.release()


epic = CamApp()
epic.run()