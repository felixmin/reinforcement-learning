import pygame
import random
import numpy as np
import time
import sys


class FlappyBirdGame:
    def __init__(self, render=False):
        # Screen dimensions
        self.SCREEN_WIDTH = 400
        self.SCREEN_HEIGHT = 600

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)

        # Game constants
        self.GRAVITY = 0.3
        self.FLAP_STRENGTH = -8
        self.PIPE_SPEED = 2
        self.PIPE_GAP = 250
        self.FPS = 60

        # Rendering setup
        self.render = render
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        else:
            self.screen = None
            self.clock = None
            self.font = None

    def close(self):
        if self.render:
            pygame.quit()

    class Bird:
        def __init__(self, game):
            self.game = game
            self.x = 50
            self.y = game.SCREEN_HEIGHT // 2
            self.radius = 15
            self.velocity = 0

        def flap(self):
            self.velocity = self.game.FLAP_STRENGTH

        def move(self):
            self.velocity += self.game.GRAVITY
            self.y += self.velocity

        def draw(self):
            if self.game.render:
                pygame.draw.circle(self.game.screen, self.game.RED, (self.x, int(self.y)), self.radius)

    class Pipe:
        def __init__(self, game, x):
            self.game = game
            self.x = x
            self.top_height = random.randint(150, game.SCREEN_HEIGHT - game.PIPE_GAP - 150)
            self.bottom_height = game.SCREEN_HEIGHT - self.top_height - game.PIPE_GAP

        def move(self):
            self.x -= self.game.PIPE_SPEED

        def draw(self):
            if self.game.render:
                pygame.draw.rect(self.game.screen, self.game.GREEN, (self.x, 0, 50, self.top_height))
                pygame.draw.rect(
                    self.game.screen, self.game.GREEN,
                    (self.x, self.game.SCREEN_HEIGHT - self.bottom_height, 50, self.bottom_height)
                )

        def collide(self, bird):
            if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + 50:
                if bird.y - bird.radius < self.top_height or bird.y + bird.radius > self.game.SCREEN_HEIGHT - self.bottom_height:
                    return True
            return False

    class FlappyBirdEnv:
        def __init__(self, game):
            self.game = game
            self.bird = None
            self.pipes = None
            self.score = None
            self.done = None
            self.reset()

        def reset(self):
            self.bird = self.game.Bird(self.game)
            self.pipes = [self.game.Pipe(self.game, self.game.SCREEN_WIDTH + 200)]
            self.score = 0
            self.done = False
            return self._get_state()

        def step(self, action):
            reward = 0.5  # Reward for survival
            if action == 1:
                self.bird.flap()
            self.bird.move()

            # Move pipes and check collisions
            for pipe in self.pipes:
                pipe.move()
                if pipe.collide(self.bird):
                    self.done = True
                    reward = -1  # Penalty for crashing

            # Remove pipes that go out of screen
            self.pipes = [pipe for pipe in self.pipes if pipe.x + 50 > 0]

            # Add new pipes if needed
            if not self.pipes or self.pipes[-1].x < self.game.SCREEN_WIDTH - 250:
                self.pipes.append(self.game.Pipe(self.game, self.game.SCREEN_WIDTH))
                self.score += 1
                reward += 1  # Reward for passing a pipe

            # Check if bird hits the ground or flies out of bounds
            if self.bird.y - self.bird.radius < 0 or self.bird.y + self.bird.radius > self.game.SCREEN_HEIGHT:
                self.done = True
                reward = -1  # Penalty for going out of bounds

            return self._get_state(), np.clip(reward, -1, 1), self.done, {}

        def render(self):
            if not self.game.render:
                return
            pygame.event.pump()  # Keep pygame responsive
            self.game.screen.fill(self.game.WHITE)
            self.bird.draw()
            for pipe in self.pipes:
                pipe.draw()
            score_text = self.game.font.render(f"Score: {self.score}", True, self.game.BLACK)
            self.game.screen.blit(score_text, (10, 10))
            pygame.display.flip()
            self.game.clock.tick(self.game.FPS)  # Maintain FPS

        def close(self):
            self.game.close()

        def _get_state(self):
            if self.pipes:
                pipe = self.pipes[0]
                return np.array([self.bird.y, self.bird.velocity, pipe.x, pipe.top_height, pipe.bottom_height])
            return np.array([self.bird.y, self.bird.velocity, self.game.SCREEN_WIDTH, 0, 0])

    def run_fn(self, test_fn):
        env = self.FlappyBirdEnv(self)
        test_fn(env)

    def play(self):
        bird = self.Bird(self)
        pipes = [self.Pipe(self, self.SCREEN_WIDTH + 200)]
        score = 0
        running = True

        while running:
            self.screen.fill(self.WHITE)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    bird.flap()

            # Bird movement
            bird.move()

            # Pipe movement and collision
            for pipe in pipes:
                pipe.move()
                if pipe.collide(bird):
                    running = False
                pipe.draw()

            # Remove pipes that go out of the screen
            pipes = [pipe for pipe in pipes if pipe.x + 50 > 0]

            # Add new pipes if needed
            if not pipes or pipes[-1].x < self.SCREEN_WIDTH - 250:
                pipes.append(self.Pipe(self, self.SCREEN_WIDTH))
                score += 1

            # Check if bird hits the ground or flies out of bounds
            if bird.y - bird.radius < 0 or bird.y + bird.radius > self.SCREEN_HEIGHT:
                running = False

            # Draw bird and update screen
            bird.draw()

            # Display score
            score_text = self.font.render(f"Score: {score}", True, self.BLACK)
            self.screen.blit(score_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.FPS)

        self.close()
        sys.exit()
