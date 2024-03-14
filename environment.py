from typing import Optional

import numpy as np
from gymnasium import spaces
import pufferlib

from pokemonred_puffer.environment import (
    RedGymEnv,
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    PARTY_SIZE,
    PARTY_LEVEL_ADDRS,
)

MUSEUM_TICKET = (0xD754, 0)


class CustomRewardEnv(RedGymEnv):
    def __init__(self, env_config: pufferlib.namespace, reward_config: pufferlib.namespace):
        super().__init__(env_config)
        self.event_obs = np.zeros(320, dtype=np.uint8)
        self._reset_reward_vars()

        # NOTE: these are not yet used
        # self.explore_weight = reward_config["explore_weight"]
        # self.explore_npc_weight = reward_config["explore_npc_weight"]
        # self.explore_hidden_obj_weight = reward_config["explore_hidden_obj_weight"]

        # NOTE: observation space must match the policy input
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
                ),
                # Discrete is more apt, but pufferlib is slower at processing Discrete
                "num_badge": spaces.Box(low=0, high=8, shape=(1,), dtype=np.uint8),
                "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                "under_limited_reward": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "cut_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            }
        )

    def _get_obs(self):
        return {
            "screen": self._get_screen_obs(),
            "num_badge": np.array(self.get_badges(), dtype=np.uint8),
            "direction": np.array(self.pyboy.get_memory_value(0xC109) // 4, dtype=np.uint8),
            "under_limited_reward": np.array(self.use_limited_reward, dtype=np.uint8),
            "cut_in_party": np.array(self.taught_cut, dtype=np.uint8),
        }

    def reset(self, seed: Optional[int] = None):
        self._reset_reward_vars()
        return super().reset(seed)

    def _reset_reward_vars(self):
        self.max_event_rew = 0
        self.max_level_sum = 0
        self.use_limited_reward = False
        self.limit_reward_cooldown = 0  # to prevent spamming limit reward

        self._update_event_obs()
        self.base_event_flags = self.event_obs.sum()

    def _update_event_obs(self):
        for i, addr in enumerate(range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)):
            self.event_obs[i] = self.bit_count(self.read_m(addr))

    def step(self, action):
        self.use_limited_reward = self.got_hm01_cut_but_not_learned_yet()
        if self.limit_reward_cooldown > 0:
            self.limit_reward_cooldown -= 1

        obs, rew, reset, _, info = super().step(action)

        self._update_event_obs()

        # NOTE: info is not always provided
        if "stats" in info:
            info["stats"]["under_limited_reward"] = self.use_limited_reward

        return obs, rew, reset, False, info

    # Reward is computed with update_reward(), which calls get_game_state_reward()
    def update_reward(self):

        # if has hm01 cut, then do not give normal reward until cut is learned
        if self.use_limited_reward:
            # encourage going to action bag menu with very small reward
            if self.seen_action_bag_menu is True and self.limit_reward_cooldown == 0:
                self.limit_reward_cooldown = 30
                return 0.0001

            return 0

        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def got_hm01_cut_but_not_learned_yet(self):
        # prev events that need to be true
        prev_events = [
            self.read_bit(0xD7F1, 0),  # met bill
            self.read_bit(0xD7F2, 3),  # used cell separator on bill
            self.read_bit(0xD7F2, 4),  # ss ticket
            self.read_bit(0xD7F2, 5),  # met bill 2
            self.read_bit(0xD7F2, 6),  # bill said use cell separator
            self.read_bit(0xD7F2, 7),  # left bills house after helping
        ]

        got_hm01 = self.read_bit(0xD803, 0)
        rubbed_captain = self.read_bit(0xD803, 1)

        return all(prev_events) and got_hm01 and rubbed_captain and not self.taught_cut

    # TODO: make the reward weights configurable
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        return {
            "event": 4 * self.update_max_event_rew(),
            "explore_npcs": sum(self.seen_npcs.values()) * 0.02,
            # "seen_pokemon": sum(self.seen_pokemon) * 0.000010,
            # "caught_pokemon": sum(self.caught_pokemon) * 0.000010,
            "moves_obtained": sum(self.moves_obtained) * 0.00010,
            "explore_hidden_objs": sum(self.seen_hidden_objs.values()) * 0.02,
            "level": self.get_levels_reward(),
            "opponent_level": self.max_opponent_level * 0.5,
            "party_size": self.party_size * 0.2,
            # "death_reward": self.died_count,
            "badge": self.get_badges() * 5,
            #"heal": self.total_heal_health,
            "explore": sum(self.seen_coords.values()) * 0.01,
            # "explore_maps": np.sum(self.seen_map_ids) * 0.0001,
            "taught_cut": 4 * int(self.taught_cut),
            "cut_coords": sum(self.cut_coords.values()) * 1.0,
            "cut_tiles": len(self.cut_tiles) * 1.0,
        }

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            self.event_obs.sum()
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )

    def get_levels_reward(self, level_cap=15):
        party_levels = [
            x for x in [self.read_m(addr) for addr in PARTY_LEVEL_ADDRS[:self.party_size]] if x > 0
        ]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))

        if self.max_level_sum < level_cap:
            return self.max_level_sum
        else:
            return level_cap + (self.max_level_sum - level_cap) / 4
