# -*- coding: utf-8 -*-
"""
DESCRIPTION

Created at 2022-08-20
Current project: OfflineScreenControlSharing


"""

import pytest

import src.hand_gesture_controller as hgc


@pytest.fixture
def controller():
    # mock_opencv()
    return hgc.HandGestureController()


class TestHandGestureController:

    def test_start_hand_gesture_recognition(self, controller):
        pass