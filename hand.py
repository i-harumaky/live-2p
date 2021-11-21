import cv2
import mediapipe as mp
import time

finger_tip_ids = [4, 8, 12, 16, 20]

class Hand():
    def __init__(
        self, 
        static_image_mode=False,
        max_hands = 2,
        detection_confidence = 0.7,
        tracking_confidence = 0.5
    ):
        self.static_image_mode = static_image_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_hands_instance = self.mp_hands.Hands(
            static_image_mode, max_hands, 
            detection_confidence, tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

        self.hand_detected = False
        self.multi_hand_landmarks = []
        self.multi_handedness = []

    def parseHands(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mp_hands_instance.process(image)
        self.multi_hand_landmarks = results.multi_hand_landmarks
        self.hand_detected = self.multi_hand_landmarks is not None
        self.multi_handedness = results.multi_handedness

    # インスタンスに保存されているネスト情報を入力した画像に書き込む
    def draw(self, image):
        if self.hand_detected:
            for hand_landmarks in self.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            return True
        return False


    # 特徴点の画像上の座標を得る
    def getAllPosition(self, image):
        landmark_coords = []
        if self.hand_detected:
            for hand_landmarks in self.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    height, width, _ = image.shape
                    cx, cy = int(landmark.x*width), int(landmark.y*height)
                    landmark_coords.append([cx, cy, landmark.z])

            return landmark_coords
        return False

    # [1,0,0,0,0] -> thumbs up
    # Left hand
    def getFingerBins(self, landmarks, is_left=True):
        result = [0,0,0,0,0]
        # check if self.hand_detected ?

        # 親指だけ左右で判定違う
        if is_left:
            if landmarks[finger_tip_ids[0]][0] > landmarks[finger_tip_ids[0] - 1][0]:
                result[0] = 1 
        else:
            if landmarks[finger_tip_ids[0]][0] < landmarks[finger_tip_ids[0] - 1][0]:
                result[0] = 1
        for idx in range(1, 5):
            if landmarks[finger_tip_ids[idx]][1] < landmarks[finger_tip_ids[idx]-2][1]:
                result[idx] = 1

        return result
