# -*- coding: utf-8 -*-

import copy
import os
import queue
import threading
import traceback
import logging

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui

import src.command_executor as com_exe
from external.Kazuhito00.app_screen_control import calc_landmark_list, pre_process_landmark
from external.Kazuhito00.model import KeyPointClassifier

VIDEO_DEVICE_ID = 0
(VIDEO_WIDTH, VIDEO_HEIGHT) = pyautogui.size()   # This gives the size of the main screen. N.B., the returned size
                                                # might be smaller than the actual size if the user zooms in the screen
                                                # size in the Windows settings
VIDEO_WIDTH = int(VIDEO_WIDTH / 3)       # Reduce the size of the video that is shown on the display
VIDEO_HEIGHT = int(VIDEO_HEIGHT / 3)

HAND_GESTURE_MODEL_PATH = './external/Kazuhito00/model/keypoint_classifier/keypoint_classifier_screen_control.tflite'

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
hand_draw_q = queue.Queue()


def draw_hand_results(results, image, hand_dict: dict) -> bool:  # pragma: no cover
    """
    Draw the hand results computed by mediapipe, and return a False flag if the user pressed ESC to quite the program.
    The hand drawing is based on midiapipe's sample code, which can be found in the "external" dir.
    """
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    left_hand_id = None
    right_hand_id = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
        left_hand_id = hand_dict.get('left').get('id')
        right_hand_id = hand_dict.get('right').get('id')
    cv.putText(image, 'Left hand ID: ' + str(left_hand_id), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv.LINE_AA)
    cv.putText(image, 'Right hand ID: ' + str(right_hand_id), (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow('MediaPipe Hands', image)
    is_continue = True
    if cv.waitKey(5) & 0xFF == 27:
        is_continue = False
    return is_continue


class HandGestureController:
    """
    Load opencv to capture hands from webcam, use mediapipe to get the hand key points, and use the trained customized
    model to identify the hand id, and finally call command_executor.execute_command_based_on_hand_signs.
    To reduce the latency, the plotting is executed in a separate thread.
    """

    def __init__(self, video_device_id=VIDEO_DEVICE_ID, video_width=VIDEO_WIDTH, video_height=VIDEO_HEIGHT):
        self._videoCap = cv.VideoCapture(video_device_id)
        self._videoCap.set(cv.CAP_PROP_FRAME_WIDTH, video_width)
        self._videoCap.set(cv.CAP_PROP_FRAME_HEIGHT, video_height)
        self._hand_sign_classifier = None
        self._command_executor_instance = None
        self._continue_main_thread = True

    def start_hand_gesture_recognition(self, show_hands=False, show_faces=False):
        self._check_if_all_models_ready()
        if show_hands:
            self._start_hand_drawing_threading()
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                            max_num_hands=2) as hands:
            while self._videoCap.isOpened():
                if self._continue_main_thread is False: break
                hand_dict = None
                try:
                    mediapipe_hand_results, image = self._get_mediapipe_hand_results(hands)
                    if mediapipe_hand_results.multi_hand_landmarks is not None:
                        hand_dict \
                            = self._get_dict_of_left_right_hands_sign_ids_and_landmarks(mediapipe_hand_results, image)
                    if show_hands:
                        self._put_hands_to_hand_draw_q(show_faces, mediapipe_hand_results, image, hand_dict)
                    if hand_dict is not None:
                        self._command_executor_instance. \
                            execute_command_based_on_hand_signs(left_hand_sign_id=hand_dict.get('left').get('id'),
                                                                left_hand_landmark=hand_dict.get('left').get(
                                                                    'landmark'),
                                                                right_hand_sign_id=hand_dict.get('right').get('id'),
                                                                right_hand_landmark=hand_dict.get('right').get(
                                                                    'landmark'))

                except (KeyboardInterrupt, Exception):
                    traceback.print_exc()
                    break

        self._videoCap.release()
        cv.destroyAllWindows()
        logging.info('Process finished successfully!')

    def load_kazuhito00_hand_sign_classifier(self, model_path: str = HAND_GESTURE_MODEL_PATH):
        if not os.path.exists(model_path):
            raise Exception(model_path + ' does not exists!')
        self._hand_sign_classifier = KeyPointClassifier(model_path=model_path)

    def set_command_executor(self, command_executor: com_exe.CommandExecutor):
        self._command_executor_instance = command_executor

    def _check_if_all_models_ready(self):
        if self._hand_sign_classifier is None:
            raise Exception('Hand_sign_classifier not assigned!')
        if self._command_executor_instance is None:
            raise Exception('Command_executor_instance not assigned!')

    def _get_mediapipe_hand_results(self, hands): # pragma: no cover
        success, image = self._videoCap.read()
        if not success:
            raise Exception('Image read error!')
        image = cv.flip(image, 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False  # To improve performance
        return hands.process(image), image

    def _get_dict_of_left_right_hands_sign_ids_and_landmarks(self, mediapipe_hand_results, image) -> dict:
        """

        :param mediapipe_hand_results:
        :param image:
        :return:
        """
        hands_sign_ids_and_landmarks_dict = {'left': {
            'id': None,
            'landmark': None
        }, 'right': {
            'id': None,
            'landmark': None
        }}
        for hand_landmarks, handedness in zip(mediapipe_hand_results.multi_hand_landmarks,
                                              mediapipe_hand_results.multi_handedness):
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            hand_sign_id = self._hand_sign_classifier(pre_processed_landmark_list)
            left_or_right_str = handedness.classification[0].label[0:].lower()
            hands_sign_ids_and_landmarks_dict[left_or_right_str] = {'id': hand_sign_id,
                                                                    'landmark': hand_landmarks.landmark}
        return hands_sign_ids_and_landmarks_dict

    def _start_hand_drawing_threading(self):
        """
        Draw the hands in a separate thread.
        """
        threading.Thread(target=self._hand_drawing_threading_func, daemon=True).start()

    def _hand_drawing_threading_func(self):
        """
        This is the target function of threading.Thread
        It will get the required arguments from the threading queue hand_draw_q, and pass the arguments to
        draw_hand_results.
        It will stop the main thread if draw_hand_results() returns False
        """
        while True:
            args = hand_draw_q.get()
            if not draw_hand_results(*args):
                self._continue_main_thread = False
                break

    @staticmethod
    def _put_hands_to_hand_draw_q(show_faces, mediapipe_hand_results, image, hand_dict):
        if show_faces:
            image_to_plot = copy.deepcopy(image)
        else:
            image_to_plot = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), np.uint8)
        args = (mediapipe_hand_results, image_to_plot, hand_dict)
        hand_draw_q.put_nowait(args)
