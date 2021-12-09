import math
from functools import cached_property
from statistics import mean

import albumentations as A
import chess
import numpy as np
import torch
import torch.nn
import torchvision
from PIL import Image
from cv2 import cv2
from cv2.mat_wrapper import Mat
from torchvision.transforms import functional as F

from training.models import get_board_segmentation_model_instance, SquareClassificationModel
from .geometry import Line, Point
from .opencv_helper import four_point_transform, order_points

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# try multiple canny / HoughLinesP parameters combinations until we detect_board_corners 81 pints
DETECTOR_PARAMETERS = [
    {
        "canny": {
            "threshold1": 80,
            "threshold2": 120,
            "apertureSize": 3,
            "L2gradient": True,
        },
        "hough": {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 120,
            "minLineLength": 100,
        }
    }, {
        "canny": {
            "threshold1": 80,
            "threshold2": 120,
            "apertureSize": 3,
            "L2gradient": True,
        },
        "hough": {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 80,
            "minLineLength": 100,
        }
    }, {
        "canny": {
            "threshold1": 20,
            "threshold2": 80,
            "apertureSize": 3,
            "L2gradient": True,
        },
        "hough": {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 80,
            "minLineLength": 100,
        }
    }, {
        "canny": {
            "threshold1": 10,
            "threshold2": 110,
            "apertureSize": 3,
            "L2gradient": True,
        },
        "hough": {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 80,
            "minLineLength": 50,
        }
    }
]


class Detector:
    debug = False
    output_window_name = "Acid Chess"

    def __init__(self, debug=False):
        self.debug = debug

    @cached_property
    def board_segmentation_model(self):
        model = get_board_segmentation_model_instance(2)
        model.load_state_dict(torch.load("training/boards.model", map_location=torch_device))
        # model.cuda()
        model.eval()
        return model

    def detect_board_corners(self, image):
        self.image = self.resize_image(image)
        self.image_debug = self.image.copy()

        print("Detecting board corners...")
        self.show_image(self.image)

        # TODO
        # tensor_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        tensor_img = self.image
        tensor_img = Image.fromarray(tensor_img)
        transform = A.Compose([
            #A.Normalize((0.1, 0.1, 0.1), (0.5, 0.5, 0.5), max_pixel_value=190.0, always_apply=True)
            # max_pixel_value=190.0,
        ])
        transformed = transform(image=np.array(tensor_img))
        tensor_img = F.to_tensor(transformed["image"])

        with torch.no_grad():
            prediction = self.board_segmentation_model([tensor_img])

        if not len(prediction[0]['masks']):
            return self.image_debug, False, [], "bord mask detection failed"

        mask = prediction[0]['masks'][0, 0]
        mask = mask.mul(255).byte().cpu().numpy()
        self.show_image(mask)
        _retval, mask = cv2.threshold(mask, 110, 1, cv2.THRESH_BINARY)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.image_debug, False, [], "bord contour extraction failed"
        contour = max(contours, key=cv2.contourArea)

        contour_len = cv2.arcLength(contour, True)
        board_edges = cv2.approxPolyDP(contour, 0.05 * contour_len, True)
        board_edges = board_edges.reshape(-1, 2)
        if board_edges is None or len(board_edges) != 4:
            return self.image_debug, False, [], "bord contour extraction failed after approxPolyDP"

        # first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        board_edges = order_points(board_edges)
        board_edges[0] = (board_edges[0][0] - 15, board_edges[0][1] - 15)
        board_edges[1] = (board_edges[1][0] + 15, board_edges[1][1] - 15)
        board_edges[2] = (board_edges[2][0] + 15, board_edges[2][1] + 15)
        board_edges[3] = (board_edges[3][0] - 15, board_edges[3][1] + 15)

        if self.debug:
            for e in board_edges:
                cv2.circle(self.image_debug, (int(e[0]), int(e[1])), 5, (0, 0, 255), -1)
            self.show_image(self.image_debug)

        try:
            warped = four_point_transform(self.image.copy(), board_edges)
            self.image_debug = warped.copy()
            self.show_image(warped)
        except ValueError:
            return self.image_debug, False, board_edges, "four_point_transform failed"

        debug_img, detected, corners, msg = self.detect_squares(warped, mode="points")
        return warped, detected, board_edges, msg

    def detect_squares(self, warped, mode="squares_obj"):
        self.image_debug = warped.copy()

        for round_nr, parameter_set in enumerate(DETECTOR_PARAMETERS):
            debug_img, detected, squares, msg = self._detect_squares_with_parameter_set(warped, mode, parameter_set)
            if detected or round_nr == len(DETECTOR_PARAMETERS) - 1:
                return debug_img, detected, squares, msg
            else:
                # try next parameter
                continue

    def _detect_squares_with_parameter_set(self, warped, mode, parameter_set):
        gray = cv2.cvtColor(warped.copy(), cv2.COLOR_BGR2GRAY)
        self.show_image(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.show_image(blurred)

        adaptiveThresh = cv2.adaptiveThreshold(blurred, 1024, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1)
        # adaptiveThresh = cv2.bitwise_not(adaptiveThresh)
        self.show_image(adaptiveThresh)

        canny = cv2.Canny(adaptiveThresh, **parameter_set["canny"])
        self.show_image(canny)

        image_h, image_w = gray.shape[0:2]
        lines = cv2.HoughLinesP(canny, maxLineGap=max(image_w, image_h), **parameter_set["hough"])
        lines_horizontal = []
        lines_vertical = []
        if lines is None:
            return self.image, False, [], "no lines after HoughLinesP found"

        for line in lines:
            line = Line(Point(line[0][0], line[0][1]), Point(line[0][2], line[0][3]))
            if line.direction == Line.HORIZONTAL:
                lines_horizontal.append(line)
            else:
                lines_vertical.append(line)

        if not lines_horizontal or not lines_vertical:
            return self.image_debug, False, [], "no v or h lines"

        lines_horizontal.sort(key=lambda l: l.p1.y + l.p2.y)
        lines_vertical.sort(key=lambda l: l.p1.x + l.p2.x)

        # remove duplicates
        for lines, image_size in [(lines_horizontal, image_h), ([lines_vertical, image_w])]:
            line_prev = None
            for line in list(lines):
                if line_prev is not None:
                    distance = line.distance(line_prev)
                    # filter by distance
                    if distance < image_size / 9:
                        lines.remove(line)
                        continue
                    else:
                        line_prev = line
                else:
                    line_prev = line

        # take middle line and walk to bottom and top
        middle_line_index = len(lines_horizontal) // 2
        lines_horizontal_final = [lines_horizontal[middle_line_index]]
        distances_sum = 0

        for index in range(1, len(lines_horizontal) // 2 + 1):
            for up_down in (-1, 1):
                line_index = middle_line_index + index * up_down
                if line_index < 0 or line_index == len(lines_horizontal):
                    continue
                line = lines_horizontal[line_index]
                prev_index = 0
                if up_down == 1:
                    prev_index = -1

                # distances_sum += distance
                lines_horizontal_final.append(line)
                lines_horizontal_final.sort(key=lambda l: l.p1.y + l.p2.y)
                if len(lines_horizontal_final) == 9:
                    break
            if len(lines_horizontal_final) == 9:
                break

        if self.debug:
            for line in lines_horizontal_final:
                cv2.line(self.image_debug, (int(line.p1.x), int(line.p1.y)), (int(line.p2.x), int(line.p2.y)),
                         (255, 0, 0), 4,
                         cv2.LINE_AA)
            self.show_image(self.image_debug)

        # if len(lines_horizontal_final) != 9:
        #    return self.image_debug, False, [], "len(lines_horizontal_final) != 9"

        # take middle line and walk to left and right
        middle_line_index = len(lines_vertical) // 2
        lines_vertical_final = [lines_vertical[middle_line_index]]
        distances_sum = 0
        for index in range(1, len(lines_vertical) // 2 + 1):
            for left_right in (-1, 1):
                line_index = middle_line_index + index * left_right
                if line_index < 0 or line_index == len(lines_vertical):
                    continue

                line = lines_vertical[line_index]
                prev_index = 0
                if left_right == 1:
                    prev_index = -1

                # line_prev = lines_vertical_final[prev_index]
                # distance = line.distance(lines_vertical_final[prev_index])
                # delta_m = abs(line_prev.m - line.m)
                # print("dm", delta_m)

                # cv2.line(self.image_debug, (int(line_prev.p1.x), int(line_prev.p1.y)),
                #          (int(line_prev.p2.x), int(line_prev.p2.y)), (0, 255, 255), 4,
                #          cv2.LINE_AA)
                # self.show_image(self.image_debug)
                # cv2.line(self.image_debug, (int(line.p1.x), int(line.p1.y)), (int(line.p2.x), int(line.p2.y)),
                #          (255, 0, 0), 4,
                #          cv2.LINE_AA)
                # self.show_image(self.image_debug)
                # distances_sum += distance
                lines_vertical_final.append(line)
                lines_vertical_final.sort(key=lambda l: l.p1.x + l.p2.x)
                if len(lines_vertical_final) == 9:
                    break
            if len(lines_vertical_final) == 9:
                break

        if self.debug:
            for line in lines_vertical_final:
                cv2.line(self.image_debug, (int(line.p1.x), int(line.p1.y)), (int(line.p2.x), int(line.p2.y)),
                         (255, 0, 0), 4,
                         cv2.LINE_AA)
            self.show_image(self.image_debug)

        # if len(lines_horizontal_final) != 9:
        #    return self.image_debug, False, [], "len(lines_horizontal_final) != 9"

        square_corners = []
        for h_line in lines_horizontal_final:
            for v_line in lines_vertical_final:
                intersect = h_line.intersection(v_line)
                if intersect is None:
                    continue

                is_duplicate = False
                for d in square_corners:
                    try:
                        if math.sqrt((d.x - intersect.x) ** 2 + (d.y - intersect.y) ** 2) < 50:
                            is_duplicate = True
                            break
                    except ValueError:
                        square_corners.append(intersect)
                        break
                if not is_duplicate:
                    square_corners.append(intersect)
                else:
                    pass
                    # print("dup")

        for sc in square_corners:
            cv2.circle(self.image_debug, (int(sc.x), int(sc.y)), 5, (0, 0, 255), -1)

        self.show_image(self.image_debug)

        if len(square_corners) != 81:
            return self.image_debug, False, [], "squares not detected, len(square_corners) != 81"

        # sort corners from top to bottom, followed by left to right
        rows = [[] for _ in range(0, 9)]
        square_corners.sort(key=lambda p: p.y)
        row_index = 0
        for c in range(0, 81):
            if c > 0 and c % 9 == 0:
                row_index = row_index + 1
            rows[row_index].append(square_corners[c])

            # sort by x if all squares of a row are available
            if len(rows[row_index]) == 9:
                rows[row_index].sort(key=lambda p: p.x)

        if mode == "points":
            return self.image_debug, True, rows, "squares as points"

        # cut squares out warped
        squares = [[] for _ in range(0, 8)]
        for row_index in [row_index for row_index in range(0, 8)]:
            for col_index in range(0, 8):
                corner_pt1 = rows[row_index][col_index]
                corner_pt2 = rows[row_index + 1][col_index + 1]

                # cv2.circle(warped, (int(corner_pt1.x), int(corner_pt1.y)), 5, (0, 0, 255), -1)
                # cv2.circle(warped, (int(corner_pt2.x), int(corner_pt2.y)), 5, (0, 0, 255), -1)
                # self.show_image(warped)

                squares_img = warped[
                              int(min(max(corner_pt1.y, 0), image_h)):int(min(max(corner_pt2.y, 0), image_h)),
                              int(min(max(corner_pt1.x, 0), image_w)):int(min(max(corner_pt2.x, 0), image_w))
                              ]
                if not squares_img.any():
                    return self.image_debug, False, [], "square extraction failed"
                square = Square(squares_img, corner_pt1, corner_pt2)
                squares[row_index].append(square)

        if mode == "squares_obj":
            return self.image_debug, True, squares, "squares detected"
        else:
            raise NotImplementedError(f"Unknown mode={mode}")

    def show_image(self, image):
        if self.debug:
            cv2.imshow(self.output_window_name, image)
            cv2.setWindowProperty(self.output_window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(0)

    def resize_image(self, image):
        target_width = 1024
        height, width = image.shape[0:2]
        size = (target_width, int(target_width / width * height))
        image = cv2.resize(image, size)
        return image


class Board(chess.Board):
    squares = [[]]

    a1_corner = None

    def update_squares(self, squares):
        if self.a1_corner is None:
            raise RuntimeError("self.a1_corner is None, needs to be set before running Board.update_squares()")
        squares.sort(self.a1_corner)
        self.squares = squares

    def diff(self, squares_to_diff):
        cl_probability_mean = squares_to_diff.get_cl_probability_mean()
        #print(cl_probability_mean)
        squares_to_diff = list(squares_to_diff.get_flat())

        # king and rook moved?
        # => castling?
        king_square = self.king(self.turn)
        king_moved = squares_to_diff[king_square].cl == Square.CL_EMPTY
        rook_moved = False
        rook_squares = list(self.pieces(chess.ROOK, self.turn))
        for square in rook_squares:
            if squares_to_diff[square].cl == Square.CL_EMPTY:
                rook_moved = True
                break

        if king_moved and rook_moved:
            for move in self.generate_legal_moves():
                square_from = squares_to_diff[move.from_square]
                square_to = squares_to_diff[move.to_square]
                if move.from_square == king_square and square_to.cl != Square.CL_EMPTY  \
                        and abs(move.from_square - move.to_square) == 2 \
                        and (square_from.cl_probability / cl_probability_mean > 0.85) \
                        and (square_to.cl_probability / cl_probability_mean > 0.85):
                    return move
            return None

        # std move
        for move in self.generate_legal_moves():
            square_from = squares_to_diff[move.from_square]
            square_to = squares_to_diff[move.to_square]
            if square_from.cl == Square.CL_EMPTY and square_to.cl == self.turn \
                    and (square_from.cl_probability / cl_probability_mean > 0.85) \
                    and (square_to.cl_probability / cl_probability_mean > 0.85):
                return move


class Square:
    CL_EMPTY = -1
    CL_WHITE = chess.WHITE
    CL_BLACK = chess.BLACK

    name = None
    cl = None
    cl_probability = 0

    def __init__(self, image: Mat, pt1: Point, pt2: Point):
        self.image = image
        self.pt1 = pt1
        self.pt2 = pt2

    @property
    def image_tensor(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = torchvision.transforms.Compose([
            torchvision.transforms.Resize(100),
            torchvision.transforms.CenterCrop(95),
            torchvision.transforms.Resize(80),
            torchvision.transforms.ToTensor(),
            # Normalize(),
        ])(image)
        return image.view(1, 3, 80, 80).to(torch_device)


class Squares:
    def __init__(self, squares):
        self.squares = squares
        self._classify()

    def __getitem__(self, row_index):
        return self.squares[row_index]

    def get_flat(self):
        for row in self.squares:
            for square in row:
                yield square

    def sort(self, a1_corner):
        # TODO allow to only run once...
        if a1_corner == (0, 0):
            self.squares = [list(s) for s in zip(*self.squares)]
        elif a1_corner == (0, 7):
            for s in self.squares:
                s.reverse()
        elif a1_corner == (7, 7):
            self.squares = [list(s) for s in zip(*self.squares)]
            for s in self.squares:
                s.reverse()
            self.squares.reverse()
        elif a1_corner == (7, 0):
            self.squares = [list(s) for s in zip(*self.squares)]
            for s in self.squares:
                s.reverse()
        else:
            raise NotImplementedError(f"a1_corner={a1_corner} unknown")

        # add names
        for index, square in enumerate(self.get_flat()):
            square.name = chess.SQUARE_NAMES[index]

    @cached_property
    def square_classification_model(self):
        # TODO
        model = SquareClassificationModel()
        model.load_state_dict(torch.load('training/model_120.pth', map_location=torch_device))
        model.eval()
        return model

    @torch.no_grad()
    def _classify(self):
        # TODO: mv, mymodel
        classes = ["black", "empty", "white"]

        for square in self.get_flat():
            prediction = self.square_classification_model(square.image_tensor)

            predicted_class_id = prediction.argmax().data.item()
            predicted_class = classes[predicted_class_id]
            if predicted_class == "black":
                square.cl = square.CL_BLACK
            elif predicted_class == "white":
                square.cl = square.CL_WHITE
            elif predicted_class == "empty":
                square.cl = square.CL_EMPTY
            else:
                raise NotImplementedError(f"Unknown predicted_class={predicted_class}")

            probability = torch.nn.Softmax(dim=1)(prediction)[0] * 100
            probability = probability.int().data.cpu().numpy()[predicted_class_id]
            square.cl_probability = probability

    def get_cl_probability_mean(self):
        return mean([s.cl_probability for s in self.get_flat()])
