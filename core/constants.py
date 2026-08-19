from enum import Enum
"yönler"
RIGHT= (0,1)
LEFT= (0,-1)
DOWN= (1,0)
UP= (-1,0)

DIRECTIONS=[RIGHT,DOWN,LEFT,UP]
DIRECTION_INDEX = {direction: i for i, direction in enumerate(DIRECTIONS)}
"aksiyonlar"
TURN_RIGHT=1
TURN_LEFT=2
GO_FORWARD=0
ACTION_NUM=3

"bayraklar"
class GameResult(Enum):
    RUNNING = "running"
    WALL = "wall"
    SELF = "self"
    STARVED = "starved"
    WON = "won"