import cv2
import subprocess
import time
class Camera:
    def __init__(self):
        self.video_capture = None
        self.idx = None
        self.is_waiting = False

    def switch_camera(self, index):
        self.is_waiting = True
        if self.video_capture is not None and self.idx is not None:
            self.video_capture.release()

        self.video_capture = cv2.VideoCapture(index)
        ret, img = self.video_capture.read()
        if ret:
            self.idx = index
        else:
            self.video_capture.release()
            if self.idx is not None:
                self.video_capture = cv2.VideoCapture(self.idx)
        self.is_waiting = False
        return ret

    def list_video_devices_linux(self):
        indexes = subprocess.check_output("ls /dev/ | grep video", shell=True).splitlines()
        idx = []
        for index in indexes:
            idx.append(int(chr(int((index[5])))))
        return idx


if __name__=="__main__":
    camera = Camera()
    print(camera.switch_camera(camera.list_video_devices_linux()[0]))
    print(camera.switch_camera(camera.list_video_devices_linux()[1]))
    print(camera.switch_camera(4))