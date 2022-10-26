# -*- coding: utf-8 -*-
"""Experiment with pyautogui
    """
import time

import pyautogui
from screeninfo import get_monitors

time.sleep(1)


def press_left_mouse():
    pyautogui.mouseDown(button='left')


def press_ctrl():
    pyautogui.keyDown('ctrl')


def hold_and_move():
    press_left_mouse()
    for ind in range(5):
        pyautogui.move(20, 20)
    pyautogui.mouseUp(button='left')

    pyautogui.mouseUp(button='left')
    for ind in range(5):
        pyautogui.move(20, 20)

    # press_ctrl()
    # pyautogui.keyUp('ctrl')
    # pyautogui.scroll(100)
    # time.sleep(2)
    # pyautogui.scroll(-100)
    # pyautogui.keyUp('ctrl')


def two_monitors():
    while True:
        print(pyautogui.position())
        print(pyautogui.size())
        time.sleep(.3)
        pyautogui.move(-50, 0)
        print(pyautogui.onScreen(*pyautogui.position()))

    # pyautogui.moveTo(2880, 0)
    # print(pyautogui.position())
    # pyautogui.moveTo(-2880, 0)
    # print(pyautogui.position())
    # print(pyautogui.onScreen(*pyautogui.position()))


def scroll():
    pyautogui.scroll(0)


if __name__ == '__main__':
    # hold_and_move()
    # two_monitors()
    # for m in get_monitors():
    #     print(m.is_primary)

    scroll()
