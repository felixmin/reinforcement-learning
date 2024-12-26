from flappy_bird.game import FlappyBirdGame
from util.modes import Modes
import fire
from reinforcement_learning.train import train_fn
from reinforcement_learning.test import test_fn

def start(mode: str):
    if mode == Modes.PLAY.value:
        game = FlappyBirdGame(render=True)
        game.play()
    elif mode == Modes.TEST.value:
        game = FlappyBirdGame(render=True)
        game.run_fn(test_fn)
    elif mode == Modes.TRAIN.value:
        game = FlappyBirdGame(render=False)
        game.run_fn(train_fn)
    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":
    fire.Fire(start)
