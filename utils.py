from typing import List, Tuple
import cv2
import numpy as np
from copy import copy


def rgb2bgr(rgb: tuple) -> tuple:
    return (rgb[2], rgb[1], rgb[0])

def zAdjuster(z:float, scale:int) -> int:
    _ = int(abs(z)*scale)
    return _ if _ >= 1 else 1


class Banner:
    def __init__(self, height:int, colors:list, background:tuple=(35,35,35)):
        self.height = height
        self.num_items = len(colors)
        self.colors = colors
        self.width = self.height // self.num_items
        self.cx = self.width//2
        self.background = background
        self.image = np.zeros((self.height, self.width, 3))
        self.image += list(self.background)
        self.setCircles()

    def setCircles(self):
        for i, color in enumerate(self.colors):
            cv2.circle(self.image, (self.cx, self.cx+self.width*i),
                        int(self.width*0.4), color, cv2.FILLED)

    def select(self, target_idx):
        # unselect all
        for i in range(self.num_items):
            cv2.circle(self.image, (self.cx, self.cx+self.width*i),
                        int(self.width*0.46), self.background, 3, lineType=cv2.LINE_AA)
            
        cv2.circle(self.image, (self.cx, self.cx+self.width*target_idx),
                    int(self.width*0.46), (255,255,255), 3, lineType=cv2.LINE_AA)

