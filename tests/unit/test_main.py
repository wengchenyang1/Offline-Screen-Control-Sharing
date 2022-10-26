# -*- coding: utf-8 -*-
"""
DESCRIPTION

Created at 2022-08-13
Current project: OfflineScreenControlSharing


"""

from unittest.mock import patch, Mock

from main import HandGestureController
from main import main


@patch.object(HandGestureController, "load_kazuhito00_hand_sign_classifier")
@patch.object(HandGestureController, "set_command_executor")
@patch.object(HandGestureController, "start_hand_gesture_recognition")
def test_main(mock_start_hand_gesture_recognition,
              mock_set_command_executor,
              mock_load_kazuhito00_hand_sign_classifier):
    mock_parent = Mock()
    mock_parent.f1 = mock_load_kazuhito00_hand_sign_classifier
    mock_parent.f2 = mock_set_command_executor
    mock_parent.f3 = mock_start_hand_gesture_recognition
    call_order = []
    mock_parent.f1.side_effect = lambda *a, **kw: call_order.append(mock_parent.f1)
    mock_parent.f2.side_effect = lambda *a, **kw: call_order.append(mock_parent.f2)
    mock_parent.f3.side_effect = lambda *a, **kw: call_order.append(mock_parent.f3)
    main()
    assert call_order == [mock_parent.f1, mock_parent.f2, mock_parent.f3]
