import numpy as np

class POMDPGridworld:
    """
    Key -> Door -> Goal Gridworld (POMDP).
    Observation: local 3x3 neighborhood flattened into 9 floats.

    FIX: Randomize key/door/start each episode.
    """
    def __init__(self, size: int = 7, seed: int | None = None):
        self.size = int(size)
        self.action_space_n = 4  # Up, Down, Left, Right
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _sample_pos(self, forbidden: set[tuple[int,int]]):
        while True:
            p = (int(self.rng.integers(0, self.size)), int(self.rng.integers(0, self.size)))
            if p not in forbidden:
                return np.array(p, dtype=np.int64)

    def reset(self):
        self.goal_pos = np.array([self.size - 1, self.size - 1], dtype=np.int64)

        forbidden = {tuple(self.goal_pos)}
        self.agent_pos = self._sample_pos(forbidden)
        forbidden.add(tuple(self.agent_pos))

        self.door_pos = self._sample_pos(forbidden)
        forbidden.add(tuple(self.door_pos))

        self.key_pos = self._sample_pos(forbidden)

        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.max_steps = self.size * self.size * 2

        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((3, 3), dtype=np.float32)
        for i in range(-1, 2):
            for j in range(-1, 2):
                p = self.agent_pos + np.array([i, j], dtype=np.int64)

                if (p < 0).any() or (p >= self.size).any():
                    obs[i+1, j+1] = 4.0
                    continue

                if np.array_equal(p, self.key_pos) and not self.has_key:
                    obs[i+1, j+1] = 1.0
                elif np.array_equal(p, self.door_pos):
                    obs[i+1, j+1] = 2.0 if not self.door_open else 0.0
                elif np.array_equal(p, self.goal_pos):
                    obs[i+1, j+1] = 3.0

        return obs.flatten().astype(np.float32)

    def step(self, action: int):
        move = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[int(action)]
        new_pos = self.agent_pos + np.array(move, dtype=np.int64)

        hit_wall = False
        if (new_pos >= 0).all() and (new_pos < self.size).all():
            if np.array_equal(new_pos, self.door_pos) and not self.door_open:
                hit_wall = True
            else:
                self.agent_pos = new_pos
        else:
            hit_wall = True

        reward = -0.01
        done = False

        if np.array_equal(self.agent_pos, self.key_pos) and not self.has_key:
            self.has_key = True
            self.door_open = True
            reward += 1.0

        if np.array_equal(self.agent_pos, self.goal_pos) and self.door_open:
            reward += 10.0
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), float(reward), bool(done), {'hit_wall': hit_wall}
