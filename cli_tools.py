#!/usr/bin/env python3
import os
from glob import glob
from pathlib import Path
from tkinter import *
from PIL import ImageTk, Image
from uuid import uuid4

import click
import numpy as np
from cv2 import cv2

from lib.board import Detector, Squares, Square

OUTPUT_DIR = "training/data/squares/sortme/"


# ./cli_tools.py board-to-squares --image-path training/data/example_boards/0005.jpg --is-start-pos 1 --orientation h --black-or-white-first w
# ./cli_tools.py example-boards-square-detection


@click.group()
def cli():
    pass


@cli.command()
@click.option("--image-path", type=str, required=False)
def example_boards_square_detection(image_path):
    if not image_path:
        images = sorted(glob("training/data/example_boards/start_lineups/*.jp*g"), reverse=True)
    else:
        images = [image_path]

    detector = Detector(debug=True)
    for image_path in images:
        click.echo(image_path)
        image = cv2.imread(image_path)
        warped, detected, corners, msg = detector.detect_board_corners(image)
        debug, detected, squares, msg = detector.detect_squares(warped, mode="images")
        if detected:
            cv2.imshow(Detector.output_window_name, warped)
            cv2.waitKey()
        else:
            click.echo(f"Unable to detect_board_corners Squares: {msg}")


@cli.command()
@click.option("--image-path", type=str, required=False)
@click.option("--debug", is_flag=True, type=bool, default=False)
def board_to_squares(image_path, debug):
    if not image_path:
        images = sorted(glob("training/data/example_boards/start_lineups/*.jp*g"), reverse=True)
    else:
        images = [image_path]

    detector = Detector(debug=debug)
    for image_path in images:
        click.echo(image_path)
        image = cv2.imread(image_path)

        warped, detected, corners, msg = detector.detect_board_corners(image)
        if not detected:
            click.echo(f"Unable to detect_board_corners: {msg}")
            continue

        debug, detected, squares, msg = detector.detect_squares(warped, mode="squares_obj")
        if not detected:
            click.echo(f"Unable to detect_squares: {msg}")
            continue

        squares = Squares(squares)
        squares.sort(a1_corner=(0,0))

        for square in squares.get_flat():
            if square.cl == Square.CL_EMPTY:
                label = "empty"
            elif square.cl == Square.CL_WHITE:
                label = "white"
            elif square.cl == Square.CL_BLACK:
                label = "black"
            output_path = os.path.join(OUTPUT_DIR, label, f"{uuid4()}.jpg")
            Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
            click.echo(output_path)
            cv2.imwrite(output_path, square.image)


@cli.command()
@click.option("--video-path", type=str, required=True)
@click.option("--debug", is_flag=True, type=bool, default=False)
def video_to_images(video_path, debug):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        click.echo("Error opening video stream or file")

    i = 0
    frame_prev = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        target_width = 800
        height, width = frame.shape[0:2]
        size = (target_width, int(target_width / width * height))
        frame = cv2.resize(frame, size)

        if frame_prev is not None:
            diff = cv2.absdiff(frame, frame_prev)
            diff = diff.astype(np.uint8)
            percentage = (np.count_nonzero(diff) * 100) / diff.size
            video_name = os.path.basename(video_path).lower().replace(" ", "_")

            if i == 0 or percentage > 65:
                image_path = f"training/data/example_boards/from_videos/sortme/{video_name}-{i}.jpeg"
                click.echo(f"percentage={percentage}, writing image {image_path}...")
                cv2.imwrite(image_path, frame)

        if debug:
            cv2.imshow("debug", frame)
            cv2.setWindowProperty("debug", cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(25):
                if 0xFF == ord('q'):
                    break

        frame_prev = frame
        i += 1

    cap.release()
    cv2.destroyAllWindows()


# @cli.command()
# @click.option("--image-path", type=str, required=False)
# def hough_parameter_tuner(image_path):
#     root = Tk("Hough Tuner")
#     image = ImageTk.PhotoImage(Image.open(image_path))
#
#
#     # canny input
#     labelframe = LabelFrame(root, text="Canny")
#     label_widget = Label(labelframe, text="Child widget of the LabelFrame")
#     labelframe.pack(padx=10, pady=10, side="left")
#     label_widget.pack(side="left")
#
#     # hough lines input
#     labelframe = LabelFrame(root, text="HoughLines")
#     label_widget = Label(labelframe, text="Child widget of the LabelFrame")
#     labelframe.pack(padx=10, pady=10, side="right")
#     label_widget.pack(side="right")
#
#     image_label = Label(root, image=image)
#     image_label.pack()
#
#     root.mainloop()


@cli.command()
@click.option("--debug", is_flag=True, type=bool, default=False)
def live_video_to_images(debug):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        click.echo("Error opening video stream or file")

    i = 0
    frame_prev = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        target_width = 800
        height, width = frame.shape[0:2]
        size = (target_width, int(target_width / width * height))
        frame = cv2.resize(frame, size)

        if frame_prev is not None:
            diff = cv2.absdiff(frame, frame_prev)
            diff = diff.astype(np.uint8)
            percentage = (np.count_nonzero(diff) * 100) / diff.size

            image_path = f"training/data/example_boards/from_videos/sortme/{uuid4()}.jpeg"
            click.echo(f"percentage={percentage}, writing image {image_path}...")
            cv2.imwrite(image_path, frame)

            detector = Detector(debug=debug)
            warped, detected, corners, msg = detector.detect_board_corners(frame)
            if not detected:
                click.echo(f"Unable to detect_board_corners: {msg}")
                continue

            debug, detected, squares, msg = detector.detect_squares(warped, mode="squares_obj")
            if not detected:
                click.echo(f"Unable to detect_squares: {msg}")
                continue

            squares = Squares(squares)
            squares.sort(a1_corner=(0, 0))

            for square in squares.get_flat():
                if square.cl == Square.CL_EMPTY:
                    label = "empty"
                elif square.cl == Square.CL_WHITE:
                    label = "white"
                elif square.cl == Square.CL_BLACK:
                    label = "black"
                output_path = os.path.join(OUTPUT_DIR, label, f"{uuid4()}.jpg")
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
                click.echo(output_path)
                cv2.imwrite(output_path, square.image)
            break

        if debug:
            cv2.imshow("debug", frame)
            cv2.setWindowProperty("debug", cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(25):
                if 0xFF == ord('q'):
                    break

        frame_prev = frame
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()
