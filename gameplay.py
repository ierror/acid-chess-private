#!/usr/bin/env python3
# model = MyModel().to(device)
import numpy as np

from cv2 import cv2

from lib.board import Detector, Board, Square
from lib.opencv_helper import four_point_transform

CAMERA_NUM = 1

detector = Detector(debug=True)

# print(detector.detect_board_corners(cv2.imread("training/data/example_boards/start_lineups/0001-h-w.jpg")))
# print(detector.detect_board_corners(cv2.imread("training/data/example_boards/gameplay/0002.jpeg")))
# print(detector.detect_board_corners(cv2.imread("training/data/example_boards/gameplay/0003.jpg")))


# exit(0)

detector = Detector()

cap = cv2.VideoCapture('DSCF2111.mov')
if not cap.isOpened():
    raise RuntimeError("Error opening video stream or file")


i = 0
previous_board_image = None
board_detected = False
corners = None
skip_until = 0
detect_orientation_todo = True

board_orientation = None
top_left_piece_color = None
hand_over_board = False

board = Board()

while cap.isOpened():
    # Capture current_frame-by-current_frame
    ret, current_frame = cap.read()
    if not ret:
        break

    msg = ""

    if i == 0:
        board_image = current_frame
    elif i <= skip_until:
        pass
    elif not board_detected:
        board_image, board_detected, corners, msg = detector.detect_board_corners(current_frame)
        if not board_detected:
            skip_until += 10
            print(msg)
    else:
        skip_until += 3

        current_frame = detector.resize_image(current_frame)
        transformed = four_point_transform(current_frame, corners)
        debug, detected, squares, msg = detector.detect_squares(transformed, mode="squares_obj")

        if not detected:
            skip_until += 5
            #board_image = debug
            continue

        board_image = transformed.copy()

        # determine board orientation
        if detect_orientation_todo:
            board.update_squares(squares)
            board.classify_squares()

            # 1. horizontal or vertical?
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
                    # print(predicted_class, probability)
                    # cv2.imshow(detector.output_window_name, square.image)
                    # cv2.setWindowProperty(detector.output_window_name, cv2.WND_PROP_TOPMOST, 1)
                    # cv2.waitKey()
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

            print(f"Top left corner: {board.row_col_index_to_san(0, 0)}")

            detect_orientation_todo = False

        else:
            if previous_board_image is None:
                skip_until += 5
                previous_board_image = board_image
                continue

            current_frame_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(previous_board_image, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

            diff = frame_diff.astype(np.uint8)
            diff_perc = (np.count_nonzero(diff) * 100) / diff.size
#            print(diff_perc)

            if hand_over_board:
                print("hand over board")

                # move stopped
                if diff_perc < 60:
                    print("hand off board")
                    board.update_squares(squares)
                    board.classify_squares()

                    diff = board.move_diff()
                    print(board)
                    if diff is not None:
                       print(diff)

                    hand_over_board = False
            else:
                if diff_perc > 65:
                    print("hand over board")
                    hand_over_board = True

            skip_until += 2
            previous_board_image = board_image.copy()

        for column in squares:
            for s in column:
                board_image = cv2.circle(board_image, (int(s.pt1.x), int(s.pt1.y)), 5, (255, 0, 255), -1)
                board_image = cv2.circle(board_image, (int(s.pt2.x), int(s.pt2.y)), 5, (255, 0, 255), -1)
                #text = "+"
                #if s.cl == Square.CL_EMPTY and s.cl_probability > 80:
                #    text = "-"
                #
                #board_image = cv2.putText(board_image, text,
                #                          (int(s.pt1.x + 25), int(s.pt2.y - 25)),
                #                          cv2.FONT_ITALIC, 1, (255, 255, 0), cv2.LINE_AA)

    cv2.imshow(detector.output_window_name, board_image)
    cv2.setWindowProperty(detector.output_window_name, cv2.WND_PROP_TOPMOST, 1)

    if cv2.waitKey(25):
        if 0xFF == ord('q'):
            break
        if 0xFF == ord('r'):
            board_detected = False

    i += 1

cap.release()
cv2.destroyAllWindows()
