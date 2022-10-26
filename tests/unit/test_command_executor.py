# -*- coding: utf-8 -*-
"""
DESCRIPTION

Created at 2022-08-13
Current project: OfflineScreenControlSharing


"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.command_executor import CommandExecutorPyAutoGUI, pyautogui


@pytest.fixture
def command_exec():
    return CommandExecutorPyAutoGUI(is_debug=False)


def _mock_all_pyautogui_functions():
    pyautogui.rightClick = MagicMock()
    pyautogui.leftClick = MagicMock()
    pyautogui.click = MagicMock()
    pyautogui.press = MagicMock()
    pyautogui.hold = MagicMock()
    pyautogui.moveTo = MagicMock()
    pyautogui.scroll = MagicMock()
    pyautogui.drag = MagicMock()
    pyautogui.mouseDown = MagicMock()
    pyautogui.keyDown = MagicMock()
    pyautogui.mouseUp = MagicMock()
    pyautogui.keyUp = MagicMock()


class TestCommandExecutorPyAutoGUI:

    @patch.object(CommandExecutorPyAutoGUI, "map_command_id_to_method")
    def test___init__(self, mock_map_command_id_to_method):
        """
        Test if map_command_id_to_method is called during init
        """
        CommandExecutorPyAutoGUI(is_debug=False)
        mock_map_command_id_to_method.assert_called_once()

    def test_map_command_id_to_method(self, command_exec):
        """
        Check variable types in the command_to_function_map returned by map_command_id_to_method
        :return:
        """
        command_to_function_map = command_exec.map_command_id_to_method()
        assert type(command_to_function_map) == dict
        for k, v in command_to_function_map.items():
            assert type(k) == int
            assert type(v).__name__ == 'function' or type(v).__name__ == 'method'
        assert len(command_to_function_map.values()) == len(set(command_to_function_map.values()))  # no repeat

    def test_get_command_id_from_hand_signs(self, command_exec):
        """
        Test return value of get_command_id_from_hand_signs for some given hand sign ids.
        :return:
        """
        assert command_exec.get_command_id_from_hand_signs(left_hand_sign_id=None, right_hand_sign_id=None) is None

    def test_get_command_method_from_hand_signs_return_none(self, command_exec):
        """
        get_command_method_from_hand_signs should return None if no command id is assigned to the given hand signs
        """
        command_exec.get_command_id_from_hand_signs = MagicMock(return_value=None)
        command_exec._append_new_command_id = MagicMock()
        comm_meth = command_exec.get_command_method_from_hand_signs(left_hand_sign_id=None, right_hand_sign_id=None)
        assert comm_meth is None
        command_exec._append_new_command_id.assert_not_called()

    def test_get_command_method_from_hand_signs_return_valid_method(self, command_exec):
        """
        get_command_method_from_hand_signs returns a valid command method
        """
        mock_command_method = MagicMock()
        command_exec._command_id_to_function_map = {8888: mock_command_method}
        command_exec.get_command_id_from_hand_signs = MagicMock(return_value=8888)
        command_exec._append_new_command_id = MagicMock()
        comm_meth = command_exec.get_command_method_from_hand_signs(left_hand_sign_id=None, right_hand_sign_id=None)
        assert comm_meth is not None
        command_exec._append_new_command_id.assert_called_once()

    def test_execute_command_based_on_hand_signs__none_command_method(self, command_exec):
        """
        Make sure that _release_all_down_keys_and_mouse_buttons is called when new command occurs
        :param command_exec:
        :return:
        """
        command_exec._release_all_down_keys_and_mouse_buttons = MagicMock()
        # None command method
        mock_command_method = None
        command_exec.get_command_method_from_hand_signs = MagicMock(return_value=mock_command_method)
        command_exec.execute_command_based_on_hand_signs()
        command_exec._release_all_down_keys_and_mouse_buttons.assert_called_once()
        command_exec._release_all_down_keys_and_mouse_buttons.reset_mock()

    def test_execute_command_based_on_hand_signs__with_command_method(self, command_exec):
        """
        Make sure that _release_all_down_keys_and_mouse_buttons is called when new command occurs
        :param command_exec:
        :return:
        """
        command_exec._release_all_down_keys_and_mouse_buttons = MagicMock()
        mock_command_method = MagicMock()
        command_exec.get_command_method_from_hand_signs = MagicMock(return_value=mock_command_method)
        command_exec._run_command_based_on_activation_status= MagicMock()

        # Test 1, the command is new
        command_exec._is_new_command_different_than_previous = MagicMock(return_value=True)
        command_exec.execute_command_based_on_hand_signs()
        command_exec._release_all_down_keys_and_mouse_buttons.assert_called_once()
        command_exec._run_command_based_on_activation_status.assert_called_once()

        # Test 2, the command is not new
        command_exec._run_command_based_on_activation_status.reset_mock()
        command_exec._release_all_down_keys_and_mouse_buttons.reset_mock()
        command_exec._is_new_command_different_than_previous = MagicMock(return_value=False)
        command_exec.execute_command_based_on_hand_signs()
        command_exec._release_all_down_keys_and_mouse_buttons.assert_not_called()
        command_exec._run_command_based_on_activation_status.assert_called_once()

    def test__run_command_based_on_activation_status(self, command_exec):
        # Test 1: command status is inactive, and the incomming command is the activation command
        command_exec.is_command_active = False
        command = MagicMock()
        command.__name__ = command_exec.ACTIVATION_COMMAND_METHOD.__name__
        command_exec._run_command_based_on_activation_status(command, 1, 2, 3, 4)
        command.assert_called_with(1, 2, 3, 4)

        # Test 2: command status is inactive, and the incomming command is not the activation command
        command_exec.is_command_active = False
        command.reset_mock()
        command.__name__ = "foo"
        command_exec._run_command_based_on_activation_status(command, 1, 2, 3, 4)
        command.assert_not_called()

        # Test 4: command status is active
        command_exec.is_command_active = True
        command.reset_mock()
        command.__name__ = "foo"
        command_exec._run_command_based_on_activation_status(command, 1, 2, 3, 4)
        command.assert_called_with(1, 2, 3, 4)


    def test__denormalize_x_y(self, command_exec):
        """
        Now matter how many screen we have, the denormalized x y, (x_d, y_d), should be non-negative integers
        TODO Mock 2 screens
        :return:
        """
        for x_n in np.linspace(0, 1, 5):
            for y_n in np.linspace(0, 1, 5):
                x_d, y_d = command_exec._denormalize_x_y(x_n, y_n, margin_top=0, margin_bottom=0, margin_left=0,
                                                         margin_right=0)
                assert x_d >= 0 and y_d >= 0
                assert type(x_d) == int and type(y_d) == int

    def test_denormalize_x_y__return_type(self, command_exec):
        """
        Make sure the return type is integer
        :param command_exec:
        :return:
        """

        with patch('pyautogui.onScreen', return_value=True):
            x_dn, y_dn = command_exec._denormalize_x_y(0.5, 0.5)
            assert type(x_dn) == int
            assert type(y_dn) == int

        class SecondScreen:
            width = 500
            height = 500
            x = 0
            y = 0

        with patch('pyautogui.onScreen', return_value=False):
            command_exec.second_monitor = SecondScreen()
            x_dn, y_dn = command_exec._denormalize_x_y(0.5, 0.5)
            assert type(x_dn) == int
            assert type(y_dn) == int

    def test__is_new_command_different_than_previous(self, command_exec):
        command_exec._command_id_history.clear()
        assert command_exec._is_new_command_different_than_previous() is False

        command_exec._command_id_history.append(0)
        assert command_exec._is_new_command_different_than_previous() is True

        command_exec._command_id_history.append(0)
        assert command_exec._is_new_command_different_than_previous() is False

        command_exec._command_id_history.append(1)
        assert command_exec._is_new_command_different_than_previous() is True

    @patch.object(CommandExecutorPyAutoGUI, '_compute_2d_distance', return_value=0)
    @patch.object(CommandExecutorPyAutoGUI, '_is_new_command_different_than_previous', return_value=True)
    @patch.object(CommandExecutorPyAutoGUI, '_update_xyz_history_for_single_hand_command')
    def test_pyautogui_commands(self, mock1, mock2, mock3, command_exec):
        """
        Test either _compute_2d_distance or _is_new_command_different_than_previous, or both, is called, in each
        pyautogui-control functions
        """
        _mock_all_pyautogui_functions()
        pyautogui_commands_names = [n for n in command_exec.__dir__() if 'pyautogui_' in n]
        command_exec._xyz_history.append((1, 2, 3))
        for pyautogui_commands in pyautogui_commands_names:
            getattr(command_exec, pyautogui_commands)(0, {}, 0, {})
            assert (mock1.call_count + mock2.call_count) >= 1

    def test__key_down(self, command_exec):
        """
        Sanity check of the key down case.
        :param command_exec:
        :return:
        """
        _mock_all_pyautogui_functions()
        command_exec._keyDown_keys = set()
        command_exec._key_down('a')
        pyautogui.keyDown.assert_called()
        assert 'a' in command_exec._keyDown_keys

        # No need to push the key if it has been pushed once
        pyautogui.keyDown.reset_mock()
        command_exec._key_down('a')
        pyautogui.keyDown.assert_not_called()

    def test__key_up(self, command_exec):
        """
        Sanity check of the key up case.
        :param command_exec:
        :return:
        """
        _mock_all_pyautogui_functions()
        command_exec._keyDown_keys = set()
        command_exec._key_up('a')
        pyautogui.keyUp.assert_not_called()

        # only call pyautogui.keyUp if the key is in _keyDown_keys
        pyautogui.keyUp.reset_mock()
        command_exec._keyDown_keys.add('a')
        command_exec._key_up('a')
        pyautogui.keyUp.assert_called()

    def test__append_new_command_id(self, command_exec):
        """
        Test if command_exec._command_id_history is updated as expected
        :param command_exec:
        :return:
        """
        assert command_exec._command_id_history.__len__() == 0
        command_exec._append_new_command_id(1)
        assert command_exec._command_id_history[0] == 1

    def test__release_all_down_keys_and_mouse_buttons(self, command_exec):
        """
        Test if key and mouse commands are called
        :param command_exec:
        :return:
        """
        command_exec._keyDown_keys = {'a'}
        command_exec._left_mouse_up = MagicMock()
        command_exec._key_up = MagicMock()

        command_exec._release_all_down_keys_and_mouse_buttons()
        command_exec._left_mouse_up.assert_called()
        command_exec._key_up.assert_called()

    def test__left_mouse_down(self, command_exec):
        _mock_all_pyautogui_functions()
        command_exec._is_left_mouse_down = True
        command_exec._left_mouse_down()
        pyautogui.mouseDown.assert_not_called()
        assert command_exec._is_left_mouse_down is True

        command_exec._is_left_mouse_down = False
        pyautogui.mouseDown.reset_mock()
        command_exec._left_mouse_down()
        pyautogui.mouseDown.assert_called()
        assert command_exec._is_left_mouse_down is True

    def test__left_mouse_up(self, command_exec):
        command_exec._is_left_mouse_down = True
        command_exec._left_mouse_up()
        pyautogui.mouseUp.assert_called()
        assert command_exec._is_left_mouse_down is False

        command_exec._is_left_mouse_down = False
        pyautogui.mouseUp.reset_mock()
        command_exec._left_mouse_up()
        pyautogui.mouseUp.assert_not_called()
        assert command_exec._is_left_mouse_down is False

    def test__is_single_hand_command(self, command_exec):
        assert command_exec._is_single_hand_command(None, 1) is True
        assert command_exec._is_single_hand_command(1, None) is True
        assert command_exec._is_single_hand_command(1, 1) is False

    def test_get_valid_landmark_for_single_hand_command(self, command_exec):
        left_hand_landmark = 777
        right_hand_landmark = 888
        left_hand_sign_id = 1
        right_hand_sign_id = None
        assert command_exec._get_valid_landmark_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                                        right_hand_sign_id,
                                                                        right_hand_landmark) == 777
        left_hand_sign_id = None
        right_hand_sign_id = 1
        assert command_exec._get_valid_landmark_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                                        right_hand_sign_id,
                                                                        right_hand_landmark) == 888

        left_hand_sign_id = None
        right_hand_sign_id = None
        assert command_exec._get_valid_landmark_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                                        right_hand_sign_id,
                                                                        right_hand_landmark) == None

        left_hand_sign_id = 1
        right_hand_sign_id = 1
        with pytest.raises(Exception, match=r"Not single hand command!"):
            command_exec._get_valid_landmark_for_single_hand_command(left_hand_sign_id, left_hand_landmark,
                                                                     right_hand_sign_id,
                                                                     right_hand_landmark)

    def test__update_xyz_history(self, command_exec):
        command_exec._update_xyz_history(1, 2, 3.0)
        assert command_exec._xyz_history[-1] == (1, 2, 3.0)


    def test__smooth_xyz(self, command_exec):
        min_pix = 5
        # self._xyz_history is now empty, so return as it is
        x0, y0, z0 = 1, 2, 4
        assert command_exec._smooth_xyz(x0, y0, z0, min_pix=min_pix) == (x0, y0, z0)
        command_exec._xyz_history.append((x0, y0, z0))

        # Now we add some pix, with only z moving with dz > min_pix
        x, y, z = x0+min_pix-1, y0+min_pix-1, z0+min_pix+1
        assert command_exec._smooth_xyz(x, y, z, min_pix=min_pix) == (x0, y0, z)

    def test_is_command_active(self, command_exec):
        command_exec._is_command_active = False
        assert command_exec.is_command_active is False
        command_exec.is_command_active = True
        assert command_exec.is_command_active is True
        command_exec.is_command_active = True
        assert command_exec.is_command_active is True
        command_exec.is_command_active = False
        assert command_exec.is_command_active is False
        command_exec.is_command_active = False
        assert command_exec.is_command_active is False
