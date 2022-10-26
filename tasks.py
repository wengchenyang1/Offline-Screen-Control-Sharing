# -*- coding: utf-8 -*-
# Tasks that can be used by "invoke".

import subprocess

from invoke import task

# TEST_DIR = 'tests/unit'
TEST_DIR = 'tests/unit/test_command_executor.py'
# TEST_DIR = 'tests/unit/test_hand_gesture_controller.py'
COV_PATH = '.coveragerc'
SRC_DIR = 'src'


@task
def lint(_):
    cmd = f"pylint {SRC_DIR}"
    subprocess.call(cmd, shell=True)


@task
def unit_test(_):
    cmd = f"pytest {TEST_DIR} --disable-warnings --cov --cov-config={COV_PATH}"
    subprocess.call(cmd, shell=True)
