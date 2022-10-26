# -*- coding: utf-8 -*-
# Created by wengc on 10/08/2022

"""
When there are bad data, we at the moment manually remove them.
"""
import matplotlib.pyplot as plt
import numpy as np

dataset = 'keypoint.csv'

target_class_id = int(input('target_class_id: '))

try:
    with open(dataset, 'r', encoding='utf-8'):
        pass
except FileNotFoundError:
    import os
    os.chdir('./external/Kazuhito00/model/keypoint_classifier')


def extract_hand_id_and_xy_data_from_line_text(line):
    raw = line.split(',')
    current_class_id = int(raw[0])
    xy = np.array([float(x_or_y) for x_or_y in raw[1:]])
    return current_class_id, np.reshape(xy, (-1, 2))


def interactively_find_line_no_of_bad_sample(lines):
    """Plot the collected hand data one by one, and decide whether keep them or not."""
    line_no_of_bad_sample = []
    with plt.ion():
        fig = plt.figure(1)
        for lno, line in enumerate(lines):
            current_class_id, xy = extract_hand_id_and_xy_data_from_line_text(line)
            if current_class_id != target_class_id:
                continue
            plt.cla()
            plt.plot(xy[:, 0], xy[:, 1], 'o')
            plt.plot(xy[0, 0], xy[0, 1], 'r+')
            for ind in range(xy.shape[0]):
                plt.text(xy[ind, 0], xy[ind, 1], str(ind))
            plt.title(f'{dataset}. Target class id: {target_class_id}. Line no.: {lno}')
            plt.gca().axis('equal')
            plt.pause(0.0001)
            res = input('Keep record? Enter/n, or p to delete the previous record')
            if res.lower() == 'n':
                line_no_of_bad_sample.append(lno)
                print(f'Line {lno} added to delete list.')
            if res.lower() == 'p' and lno > 0:
                line_no_of_bad_sample.append(lno - 1)
                print(f'Line {lno - 1} added to delete list.')
    plt.close(fig)
    return line_no_of_bad_sample


def main():
    print(f'target_class_id: {target_class_id}')
    with open(dataset, 'r+', encoding='utf-8') as fp:
        # read an store all lines into list
        lines = fp.readlines()
        lines_copy = lines.copy()
        line_no_of_wrong_sample = interactively_find_line_no_of_bad_sample(lines)
        if len(line_no_of_wrong_sample) == 0:
            return
        fp.seek(0)
        fp.truncate()
        try:
            # start writing lines
            for number, line in enumerate(lines):
                if number not in line_no_of_wrong_sample:
                    fp.write(line)
        except (KeyboardInterrupt, Exception):
            fp.seek(0)
            fp.truncate()
            fp.writelines(lines_copy)


if __name__ == '__main__':
    main()
