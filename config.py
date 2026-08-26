BOARD_ROWS = 10
BOARD_COLS = 10
SNAKE_LENGTH = 3
HUNGER_FACTOR = 2  # açlık limiti = satır × sütun × bu sayı
"OYUN"
FOOD_REWARD = 3
DEATH_PENALTY = -1
STEP_REWARD = 0
CLOSER_REWARD = 1
FARTHER_PENALTY = -1
"EKRAN"
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
GRID_SIZE = 3
GAP = 10
"HIZ"
FPS = 60
SPEED_MULTIPLIERS = [1, 2, 4]
"EGİTİM"
# POPULATION_NUM
# MUTATION_NUM
# ELITE_COUNT
# GEN_NUM
# --- Renkler (RGB) ---
COLOR_BG = (18, 18, 22)  # pencere arka planı
COLOR_BOARD = (30, 32, 38)  # tahta zemini
COLOR_BORDER = (60, 62, 70)  # normal panel çerçevesi
COLOR_LEADER = (255, 190, 60)  # lider paneli — dikkat çeksin
COLOR_HEAD = (120, 255, 140)  # yılanın başı (açık yeşil)
COLOR_BODY = (60, 180, 90)  # gövde (koyu yeşil)
COLOR_FOOD = (235, 80, 90)  # yem (kırmızı)
COLOR_TEXT = (220, 220, 230)

# Ölmüş paneller — hepsi gri, "elendi" hissi versin
COLOR_DEAD_BG = (26, 26, 30)
COLOR_SNAKE_DEAD = (70, 70, 75)
COLOR_FOOD_DEAD = (90, 70, 72)
COLOR_TEXT_DEAD = (110, 110, 118)

STATE_SIZE = 14  # duyu vektörünün uzunluğu
HIDDEN_SIZE = 12  # gizli katman nöron sayısı

FOOD_FITNESS = 2.0

STEP_FITNESS = 0.005
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.2
ELITE_COUNT = 4

GENERATION_LIMIT = 100
TRAIN_SEED = 42
GAMES_PER_GENOME = 1
