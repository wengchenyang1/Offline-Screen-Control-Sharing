# -*- coding: utf-8 -*-

# See example from mediapipe:https://google.github.io/mediapipe/solutions/hands

import logging
import sys

import src.command_executor as com_exe
from src.hand_gesture_controller import HandGestureController

HAND_GESTURE_MODEL_PATH = 'external/Kazuhito00/model/keypoint_classifier/keypoint_classifier_screen_control.tflite'


def setup_logger():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)


def main():
    setup_logger()
    controller = HandGestureController()
    controller.load_kazuhito00_hand_sign_classifier(model_path=HAND_GESTURE_MODEL_PATH)
    controller.set_command_executor(com_exe.CommandExecutorPyAutoGUI())
    logging.info("Start hand gesture recognition now.")
    controller.start_hand_gesture_recognition(show_hands=True, show_faces=False)


if __name__ == '__main__':
    main()
