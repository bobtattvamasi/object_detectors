import numpy as np
import cv2

def mouse_drawing(event, x, y, flags, detector):
    #if detector.curr_region is None:
    #    return

    if event == cv2.EVENT_LBUTTONDOWN:
        #detector.curr_region.points.append((x, y))
        print(x,y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        #if len(detector.curr_region.points) > 0:
            detector.curr_region.points.pop()
    elif event == cv2.EVENT_MOUSEMOVE:
       # detector.next_point = (x, y)
       pass



class Region:
    def __init__(self, name=''):
        self.name = name
        self.points = []

    def clear_points(self):
        self.points = []


class BorderDetector:
    def __init__(self):
        # сделать points изначально np.ndarray чтобы каждый раз не превращать в np_points?
        self.regions = []


    def draw_regions(self, frame, points, color, thickness):

        cv2.polylines(frame, points, True, color, thickness)

        return frame

    def in_region(self, x, y, xp, yp):
        c = False
        for i in range(len(xp)):
            if (((yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])) and \
                    (x > (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i])):
                c = not c
        print(c)
        return c

    def are_rectangles_in_regions(self, rectangles):
        result = [False]*len(self.regions)
        k = 0
        for region in self.regions:
            # print(region)
            if len(region) > 0:
                np_points = np.array(region)

                # for i in range(len(rectangles)):
                result[k] = self.in_region((rectangles[0] + rectangles[2]) / 2,
                                        (rectangles[1] + rectangles[3]) / 2,
                                        np_points[:, 0],
                                        np_points[:, 1])
            k += 1
        print(result)
        return result

    def add_region(self, region):
        self.regions.append(region)

    def get_region_by_index(self, index):
        return self.regions[index]

    def remove_region_by_index(self, index):
        raise NotImplementedError("You have to implement this method! (нужно будет отделить управление регионами от самого детектора)")

    def start_selecting_region(self, region, window_id):
        self.curr_region = region
        self.is_drawing = True
        self.window_id = window_id
        cv2.namedWindow(self.window_id)
        cv2.setMouseCallback(self.window_id, mouse_drawing, param=self)

    def end_selecting_region(self):
        self.next_point = None
        self.is_drawing = False
        if len(self.curr_region.points) < 3:
            self.curr_region.clear_points()
            self.regions.remove(self.curr_region)
        cv2.destroyWindow(self.window_id)
        self.window_id = None
        self.curr_region = None

    def clear_regions(self):
        self.regions.clear()