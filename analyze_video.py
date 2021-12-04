#!/usr/bin/env python3
# model = MyModel().to(device)
import chess.engine
import chess.svg
import numpy as np
import pyttsx3
from cairosvg import svg2png
from cv2 import cv2

from lib.board import Detector, Board, Square, Squares
from lib.game import Game
from lib.opencv_helper import four_point_transform

CAMERA_NUM = 1

detector = Detector(debug=True)

# print(detector.detect_board_corners(cv2.imread("training/data/example_boards/start_lineups/0001-h-w.jpg")))
# print(detector.detect_board_corners(cv2.imread("training/data/example_boards/gameplay/0002.jpeg")))
# print(detector.detect_board_corners(cv2.imread("training/data/example_boards/gameplay/0003.jpg")))


# exit(0)

detector = Detector()

# cap = cv2.VideoCapture('DSCF2121.mov')
cap = cv2.VideoCapture('DSCF2111.mov')
if not cap.isOpened():
    raise RuntimeError("Error opening video stream or file")

frame_nr = 0
board_detected = False
corners = None
skip_until = 0
detect_orientation_todo = True
previous_board_image = None
hand_over_board_start_image = None

board_orientation = None
top_left_piece_color = None
hand_over_board = False

last_frames = []
frame_nr = -1
MAX_LAST_FRAMES = 5
debug_text = None

GAME_STATE_PATH = "./tmp/gamestate.json"

LOAD = False

if LOAD:
    # TODO: a1_corner to state => detect_orientation_todo unten raus
    game_state = Game()
    game_state.load(GAME_STATE_PATH)
    board = Board()
    board.set_board_fen(game_state.board_fen)
    print(board)
else:
    game_state = Game()
    board = Board()


while cap.isOpened():
    frame_nr += 1

    ret, current_frame = cap.read()
    if not ret:
        break

    msg = ""

    if frame_nr == 0:
        board_image = current_frame
    elif frame_nr <= skip_until:
        frame_nr += 1
        continue
    elif not board_detected:
        board_image, board_detected, corners, msg = detector.detect_board_corners(current_frame)
        if not board_detected:
            skip_until += 2
            print(msg)
    elif not detect_orientation_todo and game_state.frame and frame_nr < game_state.frame - 2:
        continue
    else:
        current_frame = detector.resize_image(current_frame)
        transformed = four_point_transform(current_frame, corners)
        debug, detected, squares, msg = detector.detect_squares(transformed, mode="squares_obj")

        if not detected:
            skip_until += 5
            continue

        board_image = transformed.copy()

        # determine board orientation
        if detect_orientation_todo:
            # 1. horizontal or vertical?
            squares = Squares(squares)
            empty_squares_cnt = 0
            for row in range(0, 8):
                for col in [2, 3, 4, 5]:
                    square = squares[row][col]
                    if square.cl == square.CL_EMPTY and square.cl_probability > 90:
                        empty_squares_cnt += 1

            print(empty_squares_cnt)
            if empty_squares_cnt >= 25:
                board_orientation = "vertical"
            else:
                board_orientation = "horizontal"

            # if board_orientation == "horizontal":
            #    raise NotImplementedError()

            white_cnt = 0
            for row in range(0, 8):
                for col in [0, 1]:
                    square = squares[row][col]
                    if square.cl == square.CL_WHITE and square.cl_probability > 90:
                        white_cnt += 1

            print(white_cnt)
            if white_cnt >= 8:
                top_left_piece_color = "white"
            else:
                top_left_piece_color = "black"

            if board_orientation == "vertical":
                if top_left_piece_color == "white":
                    board.a1_corner = (0, 0)
                else:
                    board.a1_corner = (7, 7)
            else:
                if top_left_piece_color == "white":
                    board.a1_corner = (0, 7)
                else:
                    board.a1_corner = (7, 0)

            print(f"Top left corner: {chess.SQUARE_NAMES[board.a1_corner[0] * 8 + board.a1_corner[1]]}")

            detect_orientation_todo = False
            continue

        # board orientation detected
        # if the diff off last and current frame is low, there is no hand over board
        # => check board for diffs to detect a move
        if previous_board_image is None:
            previous_board_image = board_image
            last_frames.append(board_image.copy())
            continue

        if game_state.frame == frame_nr:
            squares = Squares(squares)

        frame_diff = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        for last_frame in last_frames:
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(frame_diff, last_frame)

        # current_frame_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        # previous_frame_gray = cv2.cvtColor(previous_board_image, cv2.COLOR_BGR2GRAY)
        # frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
        frame_diff = cv2.adaptiveThreshold(frame_diff, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
        diff = frame_diff.astype(np.uint8)
        diff_perc = 100 - (np.count_nonzero(diff) * 100) / diff.size

        # cv2.imshow("debug", frame_diff)
        # cv2.setWindowProperty("debug", cv2.WND_PROP_TOPMOST, 1)
        # cv2.waitKey()

        # probabilities = [s.cl_probability for s in squares.get_flat()]
        # if min(probabilities) > 90:
        previous_board_image = board_image.copy()
        last_frames.append(board_image.copy())
        last_frames = last_frames[-MAX_LAST_FRAMES:]

        debug_text = None
        if diff_perc < 5:
            # squares = Squares(squares)

            # from uuid import uuid1
            #
            # for rows in squares:
            #     for square in rows:
            #         sub_folder = "empty"
            #         if square.cl == Square.CL_WHITE:
            #             sub_folder = "white"
            #         elif square.cl == Square.CL_BLACK:
            #             sub_folder = "black"
            #
            #         s_path = f"training/data/squares/checkme/{sub_folder}/{uuid1()}.jpeg"
            #         print(s_path)
            #         cv2.imwrite(s_path, square.image)
            #         #skip_until += 5
            #
            # raise RuntimeError("stop")
            squares = Squares(squares)
            squares.sort(board.a1_corner)
            move = board.diff(squares)
            if move is not None:
                board.push(move)
                game_state.update(frame_nr, board.board_fen())
                game_state.persist(GAME_STATE_PATH)
                print(board)
                print("")

                svg2png(bytestring=chess.svg.board(board), write_to='tmp/board_rendered.png')
                # skip_until += 2
        else:
            debug_text = str(round(diff_perc))
            # skip_until += 2

        for column in squares:
            for s in column:
                board_image = cv2.circle(board_image, (int(s.pt1.x), int(s.pt1.y)), 2, (255, 0, 255), -1)
                board_image = cv2.circle(board_image, (int(s.pt2.x), int(s.pt2.y)), 2, (255, 0, 255), -1)
                if debug_text is None:
                    debug_text = str(round(s.cl_probability))
                if s.cl == Square.CL_EMPTY:
                    board_image = cv2.putText(board_image, debug_text,
                                              (int(s.pt1.x + 25), int(s.pt2.y - 25)),
                                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 0), )
                elif s.cl == Square.CL_BLACK:
                    board_image = cv2.putText(board_image, debug_text,
                                              (int(s.pt1.x + 25), int(s.pt2.y - 25)),
                                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), )
                else:
                    board_image = cv2.putText(board_image, debug_text,
                                              (int(s.pt1.x + 25), int(s.pt2.y - 25)),
                                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), )

    cv2.imwrite("tmp/debug.png", board_image)
    cv2.imshow(detector.output_window_name, board_image)
    cv2.setWindowProperty(detector.output_window_name, cv2.WND_PROP_TOPMOST, 1)

    if cv2.waitKey(1):
        if 0xFF == ord('q'):
            break

    # frame_nr += 1

cap.release()
cv2.destroyAllWindows()
