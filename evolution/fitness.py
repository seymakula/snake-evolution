import config


def fitness(game):
    total = (game.score * config.FOOD_FITNESS) + (game.steps * config.STEP_FITNESS)
    return total