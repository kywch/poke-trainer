from typing import Optional

import numpy as np

from lib.environment import (
    RedGymEnv,
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    PARTY_SIZE,
    PARTY_LEVEL_ADDRS,
)

MUSEUM_TICKET = (0xD754, 0)


class RewardWrapper(RedGymEnv):
    def __init__(self, env_config, reward_config):
        super().__init__(env_config)

        self.explore_weight = reward_config.explore_weight
        self.explore_npc_weight = reward_config.explore_npc_weight
        self.explore_hidden_obj_weight = reward_config.explore_hidden_obj_weight

        self.step_forgetting_factor = reward_config.step_forgetting_factor
        self.forgetting_frequency = reward_config.forgetting_frequency

        self._reset_reward_vars()

    def reset(self, seed: Optional[int] = None):
        self._reset_reward_vars()
        return super().reset(seed)

    def _reset_reward_vars(self):
        self.max_event_rew = 0
        self.max_level_sum = 0
        self.base_event_flags = sum(
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
        )

    def step(self, action):
        if self.step_count % self.forgetting_frequency == 0:
            self.step_forget_explore()

        return super().step(action)

    # Reward is computed with update_reward(), which calls get_game_state_reward()
    """
    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step
    """

    # TODO: make the reward weights configurable
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        state_scores = {
            "event": 4 * self.update_max_event_rew(),
            "explore_npcs": sum(self.seen_npcs.values()) * 0.02,
            # "seen_pokemon": sum(self.seen_pokemon) * 0.000010,
            # "caught_pokemon": sum(self.caught_pokemon) * 0.000010,
            "moves_obtained": sum(self.moves_obtained) * 0.00010,
            "explore_hidden_objs": sum(self.seen_hidden_objs.values()) * 0.02,
            "level": self.get_levels_reward(),
            # "opponent_level": self.max_opponent_level,
            # "death_reward": self.died_count,
            "badge": self.get_badges() * 5,
            "heal": self.total_heal_health,
            "explore": sum(self.seen_coords.values()) * 0.01,
            "explore_maps": np.sum(self.seen_map_ids) * 0.0001,
            "taught_cut": 4 * int(self.check_if_party_has_cut()),
            "cut_coords": sum(self.cut_coords.values()) * 0.001,
        }

        return state_scores

    def step_forget_explore(self):
        self.seen_coords.update(
            (k, max(0.15, v * self.step_forgetting_factor["coords"]))
            for k, v in self.seen_coords.items()
        )
        # self.seen_global_coords *= self.step_forgetting_factor["coords"]
        self.seen_map_ids *= self.step_forgetting_factor["map_ids"]
        self.seen_npcs.update(
            (k, max(0.15, v * self.step_forgetting_factor["npc"]))
            for k, v in self.seen_npcs.items()
        )
        # self.seen_hidden_objs.update(
        #     (k, max(0.15, v * self.step_forgetting_factor["hidden_objs"]))
        #     for k, v in self.seen_hidden_objs.items()
        # )
        self.explore_map *= self.step_forgetting_factor["explore"]
        self.explore_map[self.explore_map > 0] = np.clip(
            self.explore_map[self.explore_map > 0], 0.15, 1
        )

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )

    def get_levels_reward(self):
        party_size = self.read_m(PARTY_SIZE)
        party_levels = [
            x for x in [self.read_m(addr) for addr in PARTY_LEVEL_ADDRS[:party_size]] if x > 0
        ]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            return self.max_level_sum
        else:
            return 30 + (self.max_level_sum - 30) / 4
        # return 1.0 / (1 + 1000 * abs(max(party_levels) - self.max_opponent_level))
