# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Union, Tuple

import numpy as np
import pyautogui
import screeninfo

pyautogui.FAILSAFE = False


class CommandExecutor(ABC):  # pragma: no cover
    """
    Translate the left and/or right hand sign(s) into command and execute it.
    To implement new type of command methods (i.g., other than pyautogui), inheritate this abstract class, and
    edit the inherited abstract methods and the command methods that correspond to the command IDs.
    """

    @abstractmethod
    def get_command_id_from_hand_signs(self, left_hand_sign_id=None, right_hand_sign_id=None) -> int:
        """
        Return command id based on the given hand sign id(s).
        :param left_hand_sign_id: Id of the left hand, returned by the fully-conn. NN.
        :param right_hand_sign_id:  Id of the right hand, returned by the fully-conn. NN.
        :return: command id
        """
        # Example implementation:
        example_comm_id = 0
        if left_hand_sign_id == 2 or right_hand_sign_id == 2:
            example_comm_id = 1
        return example_comm_id

    @abstractmethod
    def map_command_id_to_method(self) -> dict:
        """
        Once the user got the command id, he can design the mapping between the command ids and the actual command
        methods.
        :return: A dict {command_id: command_method}
        """
        # Example implementation:
        example_command_id_to_function_map = {
            0: self.example_comm_method_0,
            1: self.example_comm_method_1
        }
        return example_command_id_to_function_map

    @abstractmethod
    def get_command_method_from_hand_signs(self, left_hand_sign_id, right_hand_sign_id):
        """
        Combine get_command_id_from_hand_signs and map_command_id_to_method so one can get the command method directly
        from the hand signs
        :return: Command method, if any command ID has been assigned to the given hand sign(s), None otherwise.
        """
        # Example implementation:
        new_command_id = self.get_command_id_from_hand_signs(
            left_hand_sign_id, right_hand_sign_id)
        command_method = self.map_command_id_to_method().get(new_command_id)
        return command_method

    @abstractmethod
    def execute_command_based_on_hand_signs(self, left_hand_sign_id=None, left_hand_landmark=None,
                                            right_hand_sign_id=None, right_hand_landmark=None):
        """
        Call the methods:
            * get_command_id_from_hand_signs
            * map_command_id_to_method
            * command_method
        :param left_hand_sign_id:
        :param left_hand_landmark:
        :param right_hand_sign_id:
        :param right_hand_landmark:
        :return:
        """
        # Example implementation:
        new_command_id = self.get_command_id_from_hand_signs(
            left_hand_sign_id, right_hand_sign_id)
        command_method = self.map_command_id_to_method().get(new_command_id)
        command_method(left_hand_sign_id, left_hand_landmark,
                       right_hand_sign_id, right_hand_landmark)

    def example_comm_method_0(self):
        pass

    def example_comm_method_1(self):
        pass


class CommandExecutorPyAutoGUI(CommandExecutor):

    def __init__(self, is_debug=True):
        self._is_debug = is_debug
        self._mouse_move_duration = 0.0
        self._command_id_history = deque(maxlen=2)
        self._finger_distance_normalized_history = deque(maxlen=2)
        self._xyz_history = deque(maxlen=3)
        self._screenWidth, self._screenHeight = pyautogui.size()
        self._command_id_to_function_map = self.map_command_id_to_method()
        self._keyDown_keys = set()
        self._is_left_mouse_down = False
        self._is_command_active = False

        self.second_monitor = self._get_2nd_monitor()

        self.ACTIVATION_COMMAND_METHOD = self.pyautogui_activate

    def execute_command_based_on_hand_signs(self, left_hand_sign_id=None, left_hand_landmark=None,
                                            right_hand_sign_id=None, right_hand_landmark=None):
        command_method = self.get_command_method_from_hand_signs(
            left_hand_sign_id, right_hand_sign_id)

        if command_method is None:
            self._release_all_down_keys_and_mouse_buttons()
            return

        if self._is_new_command_different_than_previous():
            self._release_all_down_keys_and_mouse_buttons()
            if self._is_debug and self.is_command_active:
                msg = f'New command - {self._command_id_history[-1]}: {command_method.__name__}'
                logging.debug(msg)

        self._run_command_based_on_activation_status(command_method, left_hand_sign_id, left_hand_landmark,
                                                     right_hand_sign_id, right_hand_landmark)

    def get_command_method_from_hand_signs(self, left_hand_sign_id, right_hand_sign_id) -> Union[Callable, None]:
        new_command_id = self.get_command_id_from_hand_signs(
            left_hand_sign_id, right_hand_sign_id)
        command_method = self._command_id_to_function_map.get(new_command_id)
        if command_method is not None:
            self._append_new_command_id(new_command_id)
        return command_method

    def get_command_id_from_hand_signs(self, left_hand_sign_id=None, right_hand_sign_id=None) -> Union[int, None]:
        # See presentation_materials.pptx for details
        command_id = None
        if self._is_single_hand_command(left_hand_sign_id, right_hand_sign_id):  # pragma: no cover
            if left_hand_sign_id == 2 or right_hand_sign_id == 2:
                command_id = 0
            if left_hand_sign_id == 4 or right_hand_sign_id == 4:
                command_id = 1
            if left_hand_sign_id == 3 or right_hand_sign_id == 3:
                command_id = 3
            if left_hand_sign_id == 0 or right_hand_sign_id == 0:
                command_id = 5
            if left_hand_sign_id == 1 or right_hand_sign_id == 1:
                command_id = 6
            if left_hand_sign_id == 5 or right_hand_sign_id == 5:
                command_id = 10
        else:  # pragma: no cover
            if left_hand_sign_id == 2 and right_hand_sign_id == 2:
                command_id = 2
            if left_hand_sign_id == 1 and right_hand_sign_id == 1:
                command_id = 4
            if left_hand_sign_id == 0 and right_hand_sign_id == 0:
                command_id = 5
            if left_hand_sign_id == 1 and right_hand_sign_id == 3:
                command_id = 7
            if left_hand_sign_id == 1 and right_hand_sign_id == 4:
                command_id = 8
            if left_hand_sign_id == 4 and right_hand_sign_id == 4:
                command_id = 9
            if left_hand_sign_id == 5 and right_hand_sign_id == 5:
                command_id = 11

        return command_id

    def map_command_id_to_method(self):
        return {
            0: self.pyautogui_move_to,
            1: self.pyautogui_click,
            # 2: self.pyautogui_double_click,
            3: self.pyautogui_scroll,
            4: self.pyautogui_ctrl_drag,
            5: self.pyautogui_activate,
            6: self.pyautogui_drag,
            7: self.pyautogui_ctrl_scroll,
            8: self.pyautogui_ctrl_l,
            9: self.pyautogui_ctrl_scroll_two_hands,
            # 10: self.pyautogui_right_click,
            11: self.pyautogui_deactivate
        }

    @property
    def is_command_active(self):
        return self._is_command_active

    @is_command_active.setter
    def is_command_active(self, is_active: bool):
        if self.is_command_active is not is_active:
            self._is_command_active = is_active
            if is_active is False:
                logging.info("Gesture command deactivated!")
            else:
                logging.info("Gesture command activated!")

    def _run_command_based_on_activation_status(self, command: Callable, *args):
        """Run command only when the user has activated the control commands.

        Args:
            command (Callable): Control-command method
        """
        command_is_activation = command.__name__ == self.ACTIVATION_COMMAND_METHOD.__name__
        if self._is_command_active is False and command_is_activation is False:
            return
        command(*args)

    def _get_2nd_monitor(self) -> screeninfo.Monitor or None:
        """
        Get the secondary monitor, i.e., the first non-primary monitor
        :return:
        """
        for monitor in screeninfo.get_monitors():
            if monitor.is_primary:
                continue
            logging.warning(
                "Warning: 2nd monitor detected. To get the correct mouse position, set display scale to 100%!")
            return monitor
        return None

    def _append_new_command_id(self, new_command_id):
        self._command_id_history.append(new_command_id)

    def _key_down(self, key: str):
        if key not in self._keyDown_keys:
            pyautogui.keyDown(key)
            self._keyDown_keys.add(key)

    def _key_up(self, key: str):
        if key in self._keyDown_keys:
            pyautogui.keyUp(key)

    def _release_all_down_keys_and_mouse_buttons(self):
        for k in self._keyDown_keys:
            self._key_up(k)
        self._keyDown_keys.clear()
        self._left_mouse_up()

    def _denormalize_x_y(self, x_n, y_n, margin_top=0.2, margin_bottom=.3, margin_left=0.2, margin_right=0.2) -> Tuple[
            int, int]:
        # rescale so the pointer touch the edge before the finger does so on cv screen:

        x_scale = (x_n - margin_left) / (
            1 - margin_right - margin_left)
        y_scale = (y_n - margin_top) / (1 - margin_bottom - margin_top)
        if pyautogui.onScreen(*pyautogui.position()):
            # on the primary monitor
            x_dn = self._screenWidth * x_scale
            y_dn = self._screenHeight * y_scale
        else:
            # If it's not on screen, then it must be on a 2nd screen
            x_dn = self.second_monitor.width * x_scale + \
                np.sign(self.second_monitor.x) * self._screenWidth
            y_dn = self.second_monitor.height * y_scale + \
                np.sign(self.second_monitor.y) * self._screenHeight
        return int(x_dn), int(y_dn)

    def _left_mouse_down(self):
        if self._is_left_mouse_down is False:
            pyautogui.mouseDown(button='left')
        self._is_left_mouse_down = True

    def _left_mouse_up(self):
        if self._is_left_mouse_down is True:
            pyautogui.mouseUp(button='left')
        self._is_left_mouse_down = False

    @staticmethod
    def _is_single_hand_command(left_hand_sign_id, right_hand_sign_id):
        if left_hand_sign_id is None or right_hand_sign_id is None:
            return True
        return False

    def _is_new_command_different_than_previous(self):
        if len(self._command_id_history) == 0:
            return False
        if len(self._command_id_history) == 1:
            return True
        if self._command_id_history[-1] != self._command_id_history[-2]:
            return True
        else:
            return False

    def _get_valid_landmark_for_single_hand_command(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id,
                                                    right_hand_landmark):
        if not self._is_single_hand_command(left_hand_sign_id, right_hand_sign_id):
            raise Exception("Not single hand command!")
        if left_hand_sign_id is not None:
            return left_hand_landmark
        if right_hand_sign_id is not None:
            return right_hand_landmark
        return None

    @staticmethod
    def _compute_2d_distance(left_hand_landmark, right_hand_landmark, hand_knuckle_id) -> float:
        x_l, y_l = left_hand_landmark[hand_knuckle_id].x, left_hand_landmark[hand_knuckle_id].y
        x_r, y_r = right_hand_landmark[hand_knuckle_id].x, right_hand_landmark[hand_knuckle_id].y
        return np.sqrt((x_l - x_r) ** 2 + (y_l - y_r) ** 2)

    def _update_xyz_history(self, x: int, y: int, z: float):
        self._xyz_history.append((x, y, z))

    def _smooth_xyz(self, x: int, y: int, z: int, min_pix=5) -> Tuple[int, int, int]:
        """
        Smoothen the movement of the mouse.
        If the movement of the mouse pointer is within min_pix, then don't add the new coordinate
        """

        def __smooth_coord(w, _w):
            if abs(w - _w) < min_pix:
                return _w
            return w

        if len(self._xyz_history) > 0:
            _x, _y, _z = self._xyz_history[-1]
            x = __smooth_coord(x, _x)
            y = __smooth_coord(y, _y)
            z = __smooth_coord(z, _z)
        return x, y, z

    def _update_xyz_history_for_single_hand_command(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id,
                                                    right_hand_landmark, hand_knuckle_id,
                                                    is_smooth=True, smooth_min_pix=5,
                                                    left_or_right=None):
        if self._is_new_command_different_than_previous():
            self._xyz_history.clear()
        if left_or_right is None:
            landmark = self._get_valid_landmark_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                                        right_hand_sign_id, right_hand_landmark)
        elif left_or_right == 'left':
            landmark = left_hand_landmark
        elif left_or_right == 'right':
            landmark = right_hand_landmark

        x, y = self._denormalize_x_y(
            landmark[hand_knuckle_id].x, landmark[hand_knuckle_id].y)
        z = landmark[hand_knuckle_id].z
        if is_smooth:
            x, y, z = self._smooth_xyz(x, y, z, smooth_min_pix)

        self._update_xyz_history(x, y, z)

    def pyautogui_move_to(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id, right_hand_landmark,
                          left_or_right=None):
        hand_knuckle_id = 8
        self._update_xyz_history_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                         right_hand_sign_id, right_hand_landmark,
                                                         hand_knuckle_id, left_or_right=left_or_right)
        x, y, _ = self._xyz_history[-1]
        pyautogui.moveTo(x, y, self._mouse_move_duration)

    def pyautogui_click(self, *args):
        if self._is_new_command_different_than_previous():
            pyautogui.leftClick()

    def pyautogui_double_click(self, *args):
        if self._is_new_command_different_than_previous():
            pyautogui.doubleClick()

    def pyautogui_right_click(self, *args):
        if self._is_new_command_different_than_previous():
            pyautogui.rightClick()

    def pyautogui_scroll(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id, right_hand_landmark,
                         left_or_right=None):
        hand_knuckle_id = 8
        self._update_xyz_history_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                         right_hand_sign_id, right_hand_landmark,
                                                         hand_knuckle_id, left_or_right=left_or_right)
        x, y, _ = self._xyz_history[-1]
        scroll_factor = 5
        if len(self._xyz_history) > 1:
            dy = self._xyz_history[-2][1] - y
            pyautogui.scroll(dy * scroll_factor)

    def pyautogui_drag(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id, right_hand_landmark,
                       left_or_right=None):
        hand_knuckle_id = 5
        self._update_xyz_history_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                         right_hand_sign_id, right_hand_landmark,
                                                         hand_knuckle_id,
                                                         is_smooth=True, smooth_min_pix=4, left_or_right=left_or_right)
        if self._is_new_command_different_than_previous():
            self._left_mouse_down()
        x, y, _ = self._xyz_history[-1]
        if len(self._xyz_history) > 1:
            factor = 1.5
            inverse = -1
            dx = self._xyz_history[-2][0] - x
            dy = self._xyz_history[-2][1] - y
            if dx != 0 and dy != 0:
                pyautogui.move(inverse * dx * factor, inverse *
                               dy * factor, self._mouse_move_duration)

    def pyautogui_ctrl_l(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id, right_hand_landmark):
        if self._is_new_command_different_than_previous():
            with pyautogui.hold('ctrl'):
                pyautogui.press('l')

    def pyautogui_ctrl_scroll(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id, right_hand_landmark):
        with pyautogui.hold('ctrl'):
            self.pyautogui_scroll(left_hand_sign_id, left_hand_landmark, right_hand_sign_id, right_hand_landmark,
                                  left_or_right='right')

    def pyautogui_ctrl_scroll_two_hands(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id,
                                        right_hand_landmark):
        clicks_threshold = 10
        hand_knuckle_id = 8
        scroll_factor = 4000  # Tune this factor!!
        if self._is_new_command_different_than_previous():
            self._finger_distance_normalized_history.clear()
            self._key_down('ctrl')
        new_dist = self._compute_2d_distance(
            left_hand_landmark, right_hand_landmark, hand_knuckle_id)
        self._finger_distance_normalized_history.append(new_dist)
        if len(self._finger_distance_normalized_history) > 1:
            diff = self._finger_distance_normalized_history[-1] - \
                self._finger_distance_normalized_history[-2]
            clicks = int(diff * scroll_factor)
            if abs(clicks) > clicks_threshold:
                # with pyautogui.hold("ctrl"):
                pyautogui.scroll(clicks)
            else:
                self._finger_distance_normalized_history.pop()

    def pyautogui_ctrl_drag(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id,
                            right_hand_landmark):
        if self._is_new_command_different_than_previous():
            self._key_down('ctrl')
        self.pyautogui_drag(left_hand_sign_id, left_hand_landmark, right_hand_sign_id, right_hand_landmark,
                            left_or_right='right')

    def pyautogui_activate(self, left_hand_sign_id, left_hand_landmark, right_hand_sign_id,
                           right_hand_landmark):
        # Do nothing but waking up the hand recognition system.
        if self._is_single_hand_command(left_hand_sign_id, right_hand_sign_id):
            return
        self.is_command_active = True

    def pyautogui_deactivate(self, *args):
        self.is_command_active = False
