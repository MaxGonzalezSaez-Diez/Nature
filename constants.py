from enum import IntEnum, unique


@unique
class Actions(IntEnum):
    NOTHING = 0
    REPRODUCE = 5
    SLEEP = 6


class Dim(IntEnum):
    HEALTH_SPACE = 4
    ACTION_SPACE = 7
