import io

import json
from dataclasses import dataclass
from datetime import datetime, date


@dataclass
class Game:
    engine_color: int = None
    engine_depth: int = None
    engine_time: int = None
    engine_level: int = None
    event: str = None
    round: int = None
    site: str = None
    date: str = None
    white: str = None
    black: str = None
    result: str = None
    frame: int = None
    a1_corner: str = None
    board_fen: str = None
    timestamp: datetime = None

    def update(self, frame, board_fen):
        self.frame = frame
        self.board_fen = board_fen
        self.timestamp = str(datetime.now())

    def persist(self, file_path):
        with io.open(file_path, "w") as fh:
            json.dump(self.__dict__, fh, indent=4)

    def load(self, file_path):
        with io.open(file_path, "r") as fh:
            for key, value in json.load(fh).items():
                setattr(self, key, value)
