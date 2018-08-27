import collections
import pickle


def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance



@singleton
class GlobalVar:

    @staticmethod
    def calculate_vals():
        pass


    @staticmethod
    def ret_calib_points():

        # Read in the saved objpoints and imgpoints
        dist_pickle = pickle.load(open("../util/cam_calibration_pickle.p", "rb"))
        obj_points = dist_pickle["objpoints"]
        img_points = dist_pickle["imgpoints"]
        return obj_points, img_points

    def __init__(self):
        self.idx = 0
        self.left_fit = []
        self.right_fit = []
        self.ploty = 0
        self.src = []
        self.dst = []
        self.orig_image = []
        self.offset = 0.0
        self.left_lines = collections.deque(maxlen=25)
        self.right_lines = collections.deque(maxlen=25)
        self.line_detected = collections.deque(maxlen=10)


    def set_idx(self, idx):
        self.idx = idx

    def get_idx(self):
        return self.idx

    def set_left_fit(self, left_fit):
        self.left_fit = left_fit

    def get_left_fit(self):
        return self.left_fit

    def set_right_fit(self, right_fit):
        self.right_fit = right_fit

    def get_right_fit(self):
        return self.right_fit

    def set_ploty(self, ploty):
        self.ploty = ploty

    def get_ploty(self):
        return self.ploty

    def set_src(self, src):
        self.src = src

    def get_src(self):
        return self.src

    def set_dst(self, dst):
        self.dst = dst

    def get_dst(self):
        return self.dst

    def set_orig_image(self, orig_image):
        self.orig_image = orig_image

    def get_orig_image(self):
        return self.orig_image

    def set_offset(self, offset):
        self.offset = offset

    def get_offset(self):
        return self.offset

