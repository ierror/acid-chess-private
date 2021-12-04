#!/usr/bin/env python3
import io

import queue
import threading

import chess.svg
from cv2 import cv2

PROGRAM_NAME = "Acid Chess"
CHESSBOT = "Acid"

# Game settings
GAME = {
    "event": "Testerei",
    "round": 1,
    "site": "Küche",
    "date": "2021.11.02",
    "white": "Gregor",
    "black": "Bernhard",
}

ENGINE_COLOR = chess.BLACK  # chess.WHITE, chess.BLACK, None
if ENGINE_COLOR is not None:
    GAME = {**GAME, **{
        "engine_color": ENGINE_COLOR,
        "engine_depth": 99,
        "engine_time": 30,  # in seconds
        "engine_level": 20,  # 0-20
    }}

LOAD_EXISTING = False
DEBUG = False
CAMERA_NUM = 2

# cap = cv2.VideoCapture('tmp/16 Year Old Stands Up To Intense Trash Talker! Novice Noah vs Boston Mike [V3vqq__juJc].mp4')
cap = cv2.VideoCapture('DSCF2111.mov')

# cap = cv2.VideoCapture(CAMERA_NUM)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024*1024)
if not cap.isOpened():
    raise RuntimeError("Error opening video stream or file")


class Gameplay:
    frame_nr = -1
    board_detected = False
    corners = None
    skip_until = 0
    previous_board_image = None
    hand_over_board = False
    last_engine_move_round = -1
    last_frames = []
    MAX_LAST_FRAMES = 5
    debug_text = None
    VERSION = io.open("./VERSION").readline().strip()

    def __init__(self):
        self.frame_queue = queue.Queue()

        threading.Thread(target=self.frame_worker, daemon=True).start()

        # cap = cv2.VideoCapture('tmp/16 Year Old Stands Up To Intense Trash Talker! Novice Noah vs Boston Mike [V3vqq__juJc].mp4')
        cap = cv2.VideoCapture('DSCF2111.mov')

        # cap = cv2.VideoCapture(CAMERA_NUM)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024*1024)
        if not cap.isOpened():
            raise RuntimeError("Error opening video stream or file")

        while cap.isOpened():
            ret, current_frame = cap.read()
            if not ret:
                break
            self.frame_queue.put(current_frame)

    def frame_worker(self):
        while True:
            current_frame = self.frame_queue.get()
            self.frame_nr += 1
            if not ret:
                break

            msg = ""

            if self.frame_nr == 0:
                board_image = current_frame
            elif self.frame_nr <= self.skip_until:
                continue
            elif not self.board_detected:
                board_image, board_detected, corners, msg = detector.detect_board_corners(current_frame)
                if not board_detected:
                    self.skip_until += 2
                    print(msg)
            elif game.a1_corner and game.frame and frame_nr < game.frame - 2:
                continue
            else:
                current_frame = detector.resize_image(current_frame)
                transformed = four_point_transform(current_frame, corners)
                debug, detected, squares, msg = detector.detect_squares(transformed, mode="squares_obj")

                if not detected:
                    skip_until += 5
                    continue

                skip_until = self.frame_nr
                board_image = transformed.copy()

                # determine board orientation
                if board.a1_corner is None:
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

                    game.a1_corner = board.a1_corner
                    continue

                # board orientation detected
                if previous_board_image is None:
                    previous_board_image = board_image
                    last_frames.append(board_image.copy())
                    continue

                if game.frame == frame_nr:
                    squares = Squares(squares)

                # if the diff off last and current frame is low, there is no hand over board
                # => check board for diffs to detect a move
                frame_diff = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
                for last_frame in last_frames:
                    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                    frame_diff = cv2.absdiff(frame_diff, last_frame)

                frame_diff = cv2.adaptiveThreshold(frame_diff, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                   101, 2)
                diff = frame_diff.astype(np.uint8)
                diff_perc = 100 - (np.count_nonzero(diff) * 100) / diff.size

                previous_board_image = board_image.copy()
                last_frames.append(board_image.copy())
                last_frames = last_frames[-MAX_LAST_FRAMES:]

                debug_text = None
                moved = False
                if diff_perc < 5:
                    squares = Squares(squares)
                    squares.sort(board.a1_corner)
                    move = board.diff(squares)
                    if move is not None:
                        board.push(move)
                        moved = True
                        game.update(frame_nr, board.board_fen())
                        game.persist(GAME_STATE_PATH)
                        playsound("./351518__mh2o__chess-move-on-alabaster.wav")
                        print(board)
                        print("")

                        # write board svg
                        svg2png(bytestring=chess.svg.board(board), write_to=f'{SAVE_PATH}board_rendered.png')

                        # check?
                        if (engine and board.turn != game.engine_color) or not engine:
                            if board.is_check():
                                say("check")

                        # game over?
                        outcome = board.outcome()
                        if outcome is not None:
                            if outcome.termination == chess.Termination.CHECKMATE:
                                say("game over, checkmate!")
                            elif outcome.termination == chess.Termination.STALEMATE:
                                say("game over, stalemate!")
                            elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                                say("game over, insufficient material!")
                            elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                                say("game over, fivefold repetition!")
                            elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                                say("game over, threefold repetition!")
                            elif outcome.termination == chess.Termination.VARIANT_WIN:
                                say("game over, variant win!")
                            elif outcome.termination == chess.Termination.VARIANT_LOSS:
                                say("game over, variant loss!")
                            elif outcome.termination == chess.Termination.VARIANT_DRAW:
                                say("game over, variant draw!")
                            exit(0)

                        # write pgn
                        # TODO: move to Game()
                        pgn = chess.pgn.Game()
                        pgn.headers["ProgramName"] = PROGRAM_NAME
                        pgn.headers["ProgramVersion"] = VERSION
                        for name, value in GAME.items():
                            name = name.replace("_", " ").title().replace(" ", "")
                            pgn.headers[name] = str(value)
                            if outcome is not None:
                                if outcome.winner == chess.WHITE:
                                    pgn.headers["Result"] = "1-0"
                                else:
                                    pgn.headers["Result"] = "0-1"
                        pgn.add_line(board.move_stack)
                        print(pgn, file=io.open(f"{SAVE_PATH}/game.pgn", "w"), end="\n\n")

                    # engine move
                    if engine and (moved or len(board.move_stack) > last_engine_move_round):
                        if board.turn == game.engine_color:
                            result = engine.play(board,
                                                 chess.engine.Limit(depth=game.engine_depth, time=game.engine_time))

                            from_square = chess.SQUARE_NAMES[result.move.from_square]
                            to_square = chess.SQUARE_NAMES[result.move.to_square]
                            say(f"{from_square} to {to_square}")
                            print("engine move:", f"{from_square} to {to_square}")
                            last_engine_move_round = len(board.move_stack)

                    if not hand_over_board:
                        skip_until += 4
                    hand_over_board = False
                else:
                    hand_over_board = True
                    debug_text = str(round(diff_perc))

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

            if frame_nr % 5 == 0:
                cv2.imwrite(f"{SAVE_PATH}/debug.png", board_image)

            if DEBUG:
                cv2.imshow(detector.output_window_name, board_image)
                cv2.setWindowProperty(detector.output_window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

            self.frame_queue.task_done()


Gameplay()
