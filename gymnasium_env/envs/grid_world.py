from enum import Enum
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class PosMap(Enum):
    START = 1
    BODY = 2
    GREEN_APPLE = 3
    RED_APPLE = 4
    WALL = 5


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        # Observations are dictionaries with the agent's and
        # the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e.
        # MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32
        )

        self._agent_location = np.array([-1, -1], dtype=int)
        self._agent_body = []
        self._target_location = np.array([-1, -1], dtype=int)
        self._green_apple = [np.array([-1, -1], dtype=int),
                             np.array([-1, -1], dtype=int)]

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space`
        to the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        Uses NumPy [row, col] convention where row 0 is at the top.
        """
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1]),
            Actions.UP.value: np.array([-1, 0]),
            Actions.LEFT.value: np.array([0, -1]),
            Actions.DOWN.value: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in \
            self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is use
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self, last_move=-1):
        # Vector direction : UP, DOWN, LEFT, RIGHT
        directions = [
            self._action_to_direction[Actions.UP.value],
            self._action_to_direction[Actions.DOWN.value],
            self._action_to_direction[Actions.LEFT.value],
            self._action_to_direction[Actions.RIGHT.value]
        ]

        obs_list = []
        for d in directions:
            obs_list.extend(self.create_vision(d))
        direction_one_hot = np.zeros(4, dtype=np.float32)
        if last_move != -1:
            direction_one_hot[last_move] = 1.0

        observation = np.concatenate([
            np.array(obs_list, dtype=np.float32),
            direction_one_hot
        ])
        return observation

    def create_vision(self, direction):
        result = np.zeros(4, dtype=np.float32)

        starting = self._agent_location + direction

        for i in range(1, 11):
            # 1. Détection du MUR
            if starting[0] < 0 or starting[0] >= self.size or \
             starting[1] < 0 or starting[1] >= self.size:
                result[3] = (10 - (i - 1)) / 10
                break

            # 2. Détection de la POMME VERTE
            if result[0] == 0:
                if any(np.array_equal(starting, apple)
                       for apple in self._green_apple):
                    result[0] = (10 - (i - 1)) / 10

            # 3. Détection de la POMME ROUGE
            if result[1] == 0:
                if np.array_equal(starting, self._target_location):
                    result[1] = (10 - (i - 1)) / 10

            # 4. Détection du CORPS
            if result[2] == 0:
                if any(np.array_equal(starting, segment)
                       for segment in self._agent_body):
                    result[2] = (10 - (i - 1)) / 10
            starting = starting + direction

        return result
    # def create_vision(self, direction):
    #     result = np.zeros(self.size + 1)
    #     starting = self._agent_location
    #     for i in range(self.size + 1):
    #         if np.array_equal(starting, self._agent_location):
    #             result[i] = PosMap.START.value
    #         if np.array_equal(starting, self._target_location):
    #             result[i] = PosMap.RED_APPLE.value
    #         if np.array_equal(starting, self._green_apple[0]):
    #             result[i] = PosMap.GREEN_APPLE.value
    #         if np.array_equal(starting, self._green_apple[1]):
    #             result[i] = PosMap.GREEN_APPLE.value
    #         if starting[0] < 0 or starting[0] >= self.size or \
    #            starting[1] < 0 or starting[1] >= self.size:
    #             result[i] = PosMap.WALL.value
    #             break
    #         else:
    #             for y in self._agent_body:
    #                 if np.array_equal(starting, y):
    #                     result[i] = PosMap.BODY.value
    #                     break
    #         starting = starting + direction
    #     return result

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "nbr_snake": len(self._agent_body) + 1
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._agent_body = []

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0,
                                                       self.size,
                                                       size=2,
                                                       dtype=int)
        for i in range(2):
            self.add_snake_body()

        # We will sample the target's location randomly until
        # it does not coincide with the agent's location
        self._target_location = self._agent_location
        self._target_location = self.set_location()

        self._green_apple[0] = self._target_location
        self._green_apple[1] = self._target_location

        self._green_apple[0] = self.set_location()
        self._green_apple[1] = self.set_location()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def set_location(self):
        tmp_pos = self._agent_location
        while np.array_equal(tmp_pos, self._agent_location) or \
            np.array_equal(tmp_pos, self._target_location) or \
            np.array_equal(tmp_pos, self._green_apple[0]) or \
                np.array_equal(tmp_pos, self._green_apple[1]):
            tmp_pos = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        return tmp_pos

    def add_snake_body(self):
        if not self._agent_body:
            while True:
                rdn = random.randint(0, 3)
                direction = self._action_to_direction[rdn]
                value = np.clip(
                    self._agent_location + direction, 0, self.size - 1
                    )
                if not np.array_equal(value, self._agent_location):
                    self._agent_body.append(value)
                    break
        else:
            while True:
                rdn = random.randint(0, 3)
                direction = self._action_to_direction[rdn]
                value = np.clip(
                    self._agent_body[-1] + direction, 0, self.size - 1
                    )
                if np.array_equal(value, self._agent_body[-1]) or \
                   np.array_equal(value, self._agent_location):
                    continue
                for i in self._agent_body:
                    if np.array_equal(value, i):
                        continue
                self._agent_body.append(value)
                break

    def step(self, action):
        terminated = False
        reward = 0
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid

        newpos = self._agent_location + direction

        # Wall Hit
        if not np.array_equal(newpos, np.clip(
            self._agent_location + direction, 0, self.size - 1
        )):
            reward += -10
            terminated = True

        for i in range(len(self._agent_body)):
            # Body Hit
            if np.array_equal(newpos, self._agent_body[i]):
                reward += -10
                terminated = True
        # Update Head Snake
        tmp_pos = self._agent_location
        self._agent_location = newpos

        # Update Body Snake
        if not np.array_equal(tmp_pos, self._agent_location):
            for idx in range(len(self._agent_body)):
                new_tmp = self._agent_body[idx].copy()
                self._agent_body[idx] = tmp_pos
                tmp_pos = new_tmp

        # Detection Apple
        if np.array_equal(self._agent_location, self._target_location):
            reward += -2
            self._target_location = self.set_location()
            if self._agent_body:
                self._agent_body.pop()
            else:
                reward += -8
                terminated = True

        if np.array_equal(self._agent_location, self._green_apple[0]):
            reward += 7.2
            self.add_snake_body()
            self._green_apple[0] = self.set_location()
        if np.array_equal(self._agent_location, self._green_apple[1]):
            reward += 7.2
            self.add_snake_body()
            self._green_apple[1] = self.set_location()
        reward -= 0.02

        # An episode is done iff the agent has reached the target
        # reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs(last_move=action)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._target_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3
        )

        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self._green_apple[0][::-1] + 0.5) * pix_square_size,
            pix_square_size / 3
        )

        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self._green_apple[1][::-1] + 0.5) * pix_square_size,
            pix_square_size / 3
        )
        # Now we draw the agent

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 2
        )

        for i in self._agent_body:
            pygame.draw.rect(
                canvas,
                (0, 0, 255),
                pygame.Rect(
                    pix_square_size * i[::-1],
                    (pix_square_size, pix_square_size),
                )
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings
            # from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering
            # occurs at the predefined framerate.
            # The following line will automatically
            # add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
