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
    Camera.BACK: np.array([(8.85, 2.5, 0), (8.85, 5, 0),(8.85, 7.5, 0), (8.85, 10, 0),
                           (11.35, 2.5, 0), (11.35, 5, 0), (11.35, 7.5, 0),
                           (13.85, 2.5, 0), (13.85, 5, 0),
                           (16.35, 2.5, 0)]),
    Camera.LEFT: np.array([(0, 0, 0), (0, 24, 0), (0, 70.4, 0), (50, 38, 0), (50, 24, 0)]),
    Camera.RIGHT: np.array([(0, 0, 0), (0, 24, 0), (0, 70.4, 0), (50, 38, 0), (50, 24, 0)]),
}


def plot_points(img, points, width, height, color):
    for p in points:
        p = p.reshape(-1).tolist()
        if p[0] <= 0 or p[1] <= 0 or p[0] >= width or p[1] >= height:
            continue
        p = (int(p[0]), int(p[1]))
        cv2.circle(img, p, 2, color)


def plot_arena_axis(camera, frame, world_to_pixels):
    x_points = [(i, j, 0) for i in range(98) for j in [0, 10, 20, 30, 40, 50, 60, 68.3]]
    y_points = [(j, i, 0) for i in range(69) for j in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97]]
    z_points = [(x, y, i) for i in range(46) for x in [0, 97] for y in [0, 68.3]]

    x_pixels = [world_to_pixels(x) for x in x_points]
    y_pixels = [world_to_pixels(y) for y in y_points]
    z_pixels = [world_to_pixels(z) for z in z_points]
    height, width, _ = frame.shape
    plot_points(frame, x_pixels, width, height, (0, 0, 255))
    plot_points(frame, y_pixels, width, height, (0, 255, 0))
    plot_points(frame, z_pixels, width, height, (255, 0, 0))


def get_camera_matrix_manual(camera, frame, videos_dir, manual_points_file):
    should_continue = input('Enter yes for manual file')
    if should_continue == "yes":
        with open(manual_points_file, 'rb') as file:
            chosen_points = pickle.load(file)
            arena_points, chosen_pixels = chosen_points[camera]
            print(arena_points, chosen_pixels)
            return calculate_matrix(camera, arena_points, chosen_pixels, frame, videos_dir)

    print('choose 6 arena points with known coordinates')
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    arena_points = []
    for i in range(6):
        coordinates = input(f'enter {i+1} x,y,z').split(',')
        arena_points.append((float(coordinates[0]),float(coordinates[1]),
                             float(coordinates[2])))

    points = []
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
    return calculate_matrix(camera, arena_points, points, frame, videos_dir)


def calculate_matrix(camera, arena_points, points, frame, videos_dir):
    display_frame = frame.copy()
    cc = CoordinateCalculator(model_points=np.array(arena_points))
    world_to_pixels, r, t = cc.get_world_to_pixels_translation(camera, points)
    plot_arena_axis(camera, display_frame, world_to_pixels)
    plt.figure(figsize=(12,8))
    plt.imshow(display_frame)
    plt.show()

    arena_axis_file_name = f'arena_axis_{camera}.png'
    arena_axis_file_path = "{0}/{1}".format(videos_dir, arena_axis_file_name)
    cv2.imwrite(arena_axis_file_path, display_frame)
    return r, t


def get_camera_matrix(videos_dir, video_name, manual_points_file):
    camera = file_to_camera(video_name)
    video_file = "{0}/{1}".format(videos_dir, video_name)
    video = init_video(video_file)

    result, frame = video.read()
    if not result:
        print('problem reading from video: frame {0}, result {1}'.format(i, result))
        return

    display_frame = frame.copy()
    # try to detect chessboard in frame.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    board_x_inner_corners = 9
    board_y_inner_corners = 6
    ret, corners = cv2.findChessboardCorners(gray, (board_x_inner_corners, board_y_inner_corners), None)
    chess_corners = []
    chess_3d_corners = []
    if not ret:
        print('Chessboard was not detected, Need to do manual calibration')
        should_continue = input('Enter yes for manual calibration')
        if should_continue != "yes":
            print(f'Skipping f{camera} matrix calculation')
            return
        return get_camera_matrix_manual(camera, frame, videos_dir, manual_points_file)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('image with detected chess board', img)
        cv2.waitKey(0)

        for i in range(len(corners2)):
            c = corners2[i]
            corner = c.reshape(-1)
            chess_corners.append(tuple(corner))
            frame = cv2.circle(frame, tuple(corner.astype(int)), 1, (0,0,255),2)
            cv2.putText(frame, f'{i}', tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        marked_chess_board_file_name = f'chess_board_with_indexes_{camera}.png'
        marked_chess_board_file_path = "{0}/{1}".format(videos_dir, marked_chess_board_file_name)
        cv2.imwrite(marked_chess_board_file_path, frame)
        print('saved indexed chess board to: ', marked_chess_board_file_path)
        print('initiating calibration based on chess board coordinates:')

        board_zero_x = float(input('enter point 0 x in arena'))
        board_zero_y = float(input('enter point 0 y in arena'))

        board_row_direction = float(input('is point 0 closer then point 1 to the x zero line? enter 1 if it is, -1 otherwise'))
        board_column_direction = float(input(f'is point 0 closer then point {board_x_inner_corners} to the y zero line? enter 1 if it is, -1 otherwise'))

        # coreners are ordered row by row, left to right in every row. there are ten rows.
        print('match index with position on board')
        for j in range(board_y_inner_corners):
            for i in range(board_x_inner_corners):
                num_line = (i * board_row_direction)
                num_column = (j * board_column_direction)
                board_x = board_zero_x + 2.5 * num_line
                board_y = board_zero_y + 2.5 * num_column
                #print(f'i,j:{i},{j}: x,y: {board_x},{board_y}')
                chess_3d_corners.append((board_x, board_y, 0))

        cc = CoordinateCalculator(model_points=np.array(chess_3d_corners))
        world_to_pixels, r, t = cc.get_world_to_pixels_translation(camera, chess_corners)
        plot_arena_axis(camera, display_frame, world_to_pixels)
        plt.figure(figsize=(12,8))
        plt.imshow(display_frame)
        plt.show()
        arena_axis_file_name = f'arena_axis_{camera}.png'
        arena_axis_file_path = "{0}/{1}".format(videos_dir, arena_axis_file_name)
        cv2.imwrite(arena_axis_file_path, display_frame)

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
    parser.add_argument("-p", "--manual_points", type=str,
                        help="path of manual chosen points data")
    parser.add_argument("-t", "--timestamp", type=str,
                        help="the timestamp of the experiment")

    args = parser.parse_args()

    experiment_videos = [file for file in os.listdir(args.videos_dir) if file.find(args.timestamp) != -1]
    for video in experiment_videos:
        print(f'analyzing {video}')
        camera_matrix = get_camera_matrix(args.videos_dir, video, args.manual_points)
        print(f'done analyzing {video}')
        matrix_file_path = path.join(args.matrices_dir, f'{video.split(".")[0]}_matrix.data')
        with open(matrix_file_path, 'wb') as f:
            pickle.dump(camera_matrix, f)
            print(f'writing matric to: {matrix_file_path}')

    print('Sanity check: printing location of cameras relatively to the arena')
    arena = load_arena(args.matrices_dir, args.timestamp)
    camera_point = np.array([0, 0, 0]).reshape((3, 1))
    for c in Camera:
        p = arena.translate_point_to_world(c, camera_point)
        print(c, p)


if __name__ == '__main__':
    main()
