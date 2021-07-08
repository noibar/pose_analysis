import argparse
import cv2
import pickle
import os
from os import path
import matplotlib.pyplot as plt
import time

import numpy as np
from arena import load_arena, file_to_camera
from camera import Camera
from utils import init_video
from coordinate_calculator import CoordinateCalculator

CALIBRATION_POINTS = {
    Camera.TOP: np.array([(0, 0, 0), (0, 70.4, 0), (99, 70.4, 0), (99, 0, 0)]),
    Camera.BACK: np.array([(99, 0, 0), (99, 28, 0), (35, 70.4, 0), (35, 38, 0), (35, 24, 0), (35, 0, 0)]),
    Camera.LEFT: np.array([(0, 0, 0), (0, 24, 0), (0, 70.4, 0), (50, 38, 0), (50, 24, 0)]),
    Camera.RIGHT: np.array([(0, 0, 0), (0, 24, 0), (0, 70.4, 0), (50, 38, 0), (50, 24, 0)]),
}


def plot_points(img, points, width, height):
    for p in points:
        p = p.reshape(-1).tolist()
        if p[0] <= 0 or p[1] <= 0 or p[0] >= width or p[1] >= height:
            continue
        p = (int(p[0]), int(p[1]))
        cv2.circle(img, p, 2, (0, 0, 255))


def plot_arena_axis(camera, frame, world_to_pixels):
    x_points = [(i, 0, 0) for i in range(99)]
    if camera == Camera.BACK:
        y_points = [(99, i, 0) for i in range(70)]
        z_points = [(99, 0, i) for i in range(40)]
    else:
        y_points = [(0, i, 0) for i in range(70)]
        z_points = [(0, 0, i) for i in range(40)]
    x_pixels = [world_to_pixels(x) for x in x_points]
    y_pixels = [world_to_pixels(y) for y in y_points]
    z_pixels = [world_to_pixels(z) for z in z_points]
    height, width, _ = frame.shape
    plot_points(frame, x_pixels, width, height)
    plot_points(frame, y_pixels, width, height)
    plot_points(frame, z_pixels, width, height)


def get_camera_matrix(videos_dir, video_name):
    camera = file_to_camera(video_name)
    arena_points = CALIBRATION_POINTS[camera]
    cc = CoordinateCalculator(model_points=arena_points)
    video_file = "{0}/{1}".format(videos_dir, video_name)
    video = init_video(video_file)

    result, frame = video.read()
    if not result:
        print('problem reading from video: frame {0}, result {1}'.format(i, result))
        return

    points = []
    display_frame = frame.copy()

    cv2.imshow('frame', display_frame)
    def onclick(event, x, y, flags, params):
        if event != cv2.EVENT_RBUTTONDOWN:
            return

        points.append([x, y])
        if len(points) == len(arena_points):
            cv2.destroyAllWindows()
            return
        print('right click point {}'.format(arena_points[len(points)]))

    cv2.setMouseCallback('frame', onclick)
    print('right click point {}'.format(arena_points[len(points)]))
    cv2.waitKey(0)

    print('selected points: ', points)
    world_to_pixels, r, t = cc.get_world_to_pixels_translation(camera, points)
    plot_arena_axis(camera, display_frame, world_to_pixels)
    plt.figure(figsize=(12,8))
    plt.imshow(display_frame)
    plt.show()
    #cv2.imshow('frame', display_frame)
    #print('press enter to continue')
    #while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
    #    print('.')
    #    cv2.waitKey(50)
    return r, t


def get_extrinsic_matrix(r, t):
    extrinsic_matrix = np.zeros((3, 4), dtype=float)
    extrinsic_matrix[:3, :3] = r
    extrinsic_matrix[:, 3] = t.reshape((3,))
    return extrinsic_matrix


def camera_to_world(r, t, point):
    point = point - t
    m = np.linalg.inv(np.matrix(r))
    return m * point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--videos_dir", type=str, default=os.path.join("videos", "undistorted"),
                        help="directory of the input videos")
    parser.add_argument("-m", "--matrices_dir", type=str, default="matrices",
                        help="directory of output matrices")
    parser.add_argument("-t", "--timestamp", type=str,
                        help="the timestamp of the experiment")

    args = parser.parse_args()

    experiment_videos = [file for file in os.listdir(args.videos_dir) if file.find(args.timestamp) != -1]
    for video in experiment_videos:
        print(f'analyzing {video}')
        camera_matrix = get_camera_matrix(args.videos_dir, video)
        print(f'done analyzing {video}')
        with open(path.join(args.matrices_dir, f'{video.split(".")[0]}_matrix.data'), 'wb') as f:
            pickle.dump(camera_matrix, f)

    print('Sanity check: printing location of cameras relatively to the arena')
    arena = load_arena(args.matrices_dir, args.timestamp)
    camera_point = np.array([0, 0, 0]).reshape((3, 1))
    for c in Camera:
        p = arena.translate_point_to_world(c, camera_point)
        print(c, p)


if __name__ == '__main__':
    main()
