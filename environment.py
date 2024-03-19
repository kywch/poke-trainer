from typing import Optional

import numpy as np
from gymnasium import spaces
import pufferlib

from pokemonred_puffer.environment import (
    RedGymEnv,
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    PARTY_LEVEL_ADDRS,
)

MUSEUM_TICKET = (0xD754, 0)

MENU_COOLDOWN = 200
PRESS_BUTTON_A = 5

BASE_TILE_NUM = 500  # tile exploration reward starts from this

class CustomRewardEnv(RedGymEnv):
    def __init__(self, env_config: pufferlib.namespace, reward_config: pufferlib.namespace):
        super().__init__(env_config)
        self.init_max_steps = env_config.max_steps

        #self.event_obs = np.zeros(320, dtype=np.uint8)
        self.event_count = {}  # kept for whole training run, take this when checkpointing
        self.experienced_events = set()  # reset every episode
        self.event_reward = np.zeros(320)
        self.base_event_reward = 0

        self._reset_reward_vars()

        # NOTE: these are not yet used
        # self.explore_weight = reward_config["explore_weight"]
        # self.explore_npc_weight = reward_config["explore_npc_weight"]
        # self.explore_hidden_obj_weight = reward_config["explore_hidden_obj_weight"]

        # NOTE: decaying seen coords/tiles makes reward dense, making the place more "sticky"
        # not well understood the dynamics yet. Buy decaying the value when there are so many coords
        # decrease the summed value much and thus push the agents to visit new coord more, just to fill it
        # Thus using MaxLengthWrapper to cap the seen coords at 3000
        self.tile_reward = 0
        self.seen_tiles = {}
        self.decay_factor_coords = 0.9995
        self.decay_factor_npcs = 0.995
        self.decay_frequency = 10

        # NOTE: observation space must match the policy input
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
                ),
                # Discrete is more apt, but pufferlib is slower at processing Discrete
                # "x": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                # "y": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                "curr_map_idx": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                "map_progress": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                "num_badge": spaces.Box(low=0, high=8, shape=(1,), dtype=np.uint8),
                "party_size": spaces.Box(low=0, high=8, shape=(1,), dtype=np.uint8),
                "seen_pokemon": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),

                # TODO: probably need some level obs, for max/min/sum of the party?
                #"direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),  # TODO: replace with tree in front

                # NOTE: if there are other conditions for limited reward, more flags should be added
                "under_limited_reward": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "cut_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            }
        )

    def _get_obs(self):
        # See pokemonred_puffer/map_data.json for map ids
        player_x, player_y, map_idx = self.get_game_coords()
        map_idx += 1  # map_id starts from -1 (Kanto) to 247 (Agathas room)

        return {
            "screen": self._get_screen_obs(),
            # "x": np.array(player_x, dtype=np.uint8),
            # "y": np.array(player_y, dtype=np.uint8),
            "curr_map_idx": np.array(map_idx, dtype=np.uint8),  # using 256 one-hot
            "map_progress": np.array(self.max_map_progress, dtype=np.uint8),  # using 16 one-hot
            "num_badge": np.array(self.get_badges(), dtype=np.uint8),  # using 9 one-hot
            "party_size": np.array(self.party_size, dtype=np.uint8),  # using 8 one-hot -- CHECK ME: below 8?
            "seen_pokemon": np.array(self.seen_pokemon.sum(), dtype=np.uint8),  # a great proxy for game progression
            #"direction": np.array(self.pyboy.get_memory_value(0xC109) // 4, dtype=np.uint8),
            "under_limited_reward": np.array(self.use_limited_reward, dtype=np.uint8),
            "cut_in_party": np.array(self.taught_cut, dtype=np.uint8),
        }

    def reset(self, seed: Optional[int] = None):
        # After each reset, increase max steps
        self.max_steps += self.init_max_steps

        # Load the state and reset all the RedGymEnv vars
        obs, info = super().reset(seed)

        # NOTE: these dict are used for exploration decay within episode
        # So they should be reset every episode
        if self.first is False:
            self.seen_tiles.clear()
            self.seen_coords.clear()
            self.seen_npcs.clear()
            self.seen_hidden_objs.clear()

        self.total_reward = 0  # NOTE: super.reset() updates this before resetting reward vars
        self._reset_reward_vars()

        return obs, info

    def _reset_reward_vars(self):
        self.use_limited_reward = False

        #self.event_obs.fill(0)
        self.experienced_events.clear()
        self.event_reward.fill(0)
        self.base_event_reward = 0
        self.max_event_rew = 0
        self._update_event_obs()
        self.consumed_item_count = 0

        self.tile_reward = 0
        self.max_level_sum = 0

        # KEY events
        self.badges = 0
        self.bill_said_use_cell_separator = False
        self.got_hm01 = False

        # Track action bag menu
        self.action_bag_menu_count = 0
        self.rewarded_action_bag_menu = 0
        self.pokemon_action_count = 0
        self.rewared_pokemon_action = 0
        self.menu_reward_cooldown = 0  # to prevent spamming limit reward

        # Track learn moves with item
        self.curr_moves = 0
        self.curr_item_num = 0
        self.moves_learned_with_item = 0
        self.just_learned_item_move = 0

    def step(self, action):
        if self.menu_reward_cooldown > 0:
            self.menu_reward_cooldown -= 1

        self.use_limited_reward = self.got_hm01_cut_but_not_learned_yet()
        # if self.use_limited_reward:
        #     # NOTE: only for HM cut now
        #     self.set_cursor_to_item(target_id=0xC4)  # 0xC4: HM cut
        #     pass

        # Apply decay on the seen coords and npcs
        if self.step_count % self.decay_frequency == 0:
            self.step_decay_seen_coords()

        obs, rew, reset, _, info = super().step(action)

        self._update_menu_reward_vars(action)

        # NOTE: info is not always provided
        if "stats" in info:
            info["stats"]["under_limited_reward"] = self.use_limited_reward  # rename to hm 01, menu time?
            info["stats"]["learn_with_item"] = self.moves_learned_with_item
            info["stats"]["action_bag_menu_count"] = self.action_bag_menu_count
            info["stats"]["rewarded_action_bag_menu"] = self.rewarded_action_bag_menu
            info["stats"]["pokemon_action_count"] = self.pokemon_action_count
            info["stats"]["rewared_pokemon_action"] = self.rewared_pokemon_action
            info["stats"]["consumed_item_count"] = self.consumed_item_count

            # Does the events get correctly reset?
            # NOTE: event is not resetting to 0. REVISIT THIS
            info["stats"]["new_event_reward"] = self.event_reward.sum()
            info["stats"]["new_tile_reward"] = self.tile_reward

        return obs, rew, reset, False, info

    def _update_menu_reward_vars(self, action):
        # Check menu action
        if action is not None and action == PRESS_BUTTON_A:
            if self.check_if_in_bag_menu():
                self.action_bag_menu_count += 1
                if self.menu_reward_cooldown == 0:
                    self.rewarded_action_bag_menu += 1
                    self.menu_reward_cooldown = MENU_COOLDOWN
            if self.check_if_in_pokemon_menu():
                self.pokemon_action_count += 1
                if self.menu_reward_cooldown == 0:
                    self.rewared_pokemon_action += 1
                    self.menu_reward_cooldown = MENU_COOLDOWN

    # Reward is computed with update_reward(), which calls get_game_state_reward()
    def update_reward(self):
        # if has hm01 cut, then do not give normal reward until cut is learned
        if self.use_limited_reward:
            if self.just_learned_item_move > 0:  # encourage any learning from item
                self.moves_learned_with_item += 1
                return 0.1
            # encourage going to action bag menu with very small reward
            if self.seen_action_bag_menu == 1 and self.menu_reward_cooldown == 0:
                self.menu_reward_cooldown = 30
                self.action_bag_menu_count += 1
                self.rewarded_action_bag_menu += 1
                return 0.001
            # None of the above -- no reward
            return 0

        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])

        # Total reward is not 0 at the beginning, so make the initial reward 0
        new_step = new_total - self.total_reward if self.total_reward > 0 else 0

        self.total_reward = new_total
        return new_step

    # TODO: make the reward weights configurable
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        self._update_event_reward_vars()
        self._update_tile_reward_vars()

        # NOTE: subtract BASE_TILE_NUM to ignore the initial maps
        adj_tile_reward = max(self.tile_reward - BASE_TILE_NUM, 0) ** 0.9

        return {
            # Main milestones for story progression
            "badge": self.badges * 5.0,
            "map_progress": self.max_map_progress * 2.0,
            #"opponent_level": self.max_opponent_level * 1.0,
            "key_events": self.key_events_reward * 2.0,  # bill_said, got_hm01, taught_cut

            # Party strength proxy
            "party_size": self.party_size * 3.0,
            "level": self.get_levels_reward(),

            # Important skill: learning moves with items
            "learn_with_item": self.moves_learned_with_item * 3.0,

            # Exploration: bias agents' actions with weight for each new gain
            # These kick in when agent is "stuck"

            # NOTE: exploring "newer" tiles is the main driver of progression
            # Visit decay makes the explore reward "dense" ... little reward everywhere
            # so agents are motivated to explore new coords and/or revisit old coords
            "explore_tile": adj_tile_reward * 0.01,

            # First, always search for new pokemon and events
            "seen_pokemon": self.seen_pokemon.sum() * 1.5,  # more related to story progression?

            # NOTE: there seems to be a lot of irrevant events?
            # event weight ~0: after 1st reset, agents go straight to the next target, but after 2-3, it forgets to make progress
            # event weight 1: atter 1st reset, agents stick to "old" events, that guarantee reward ... so does not make progress
            # after seeing this, implemented the experienced event reward discounting
            "event": self.max_event_rew * 1.0,

            # If the above doesn't work, try these in the order of importance
            "explore_npcs": sum(self.seen_npcs.values()) * 0.03,  # talk to new npcs
            "explore_hidden_objs": sum(self.seen_hidden_objs.values()) * 0.02,  # look for new hidden objs
            "moves_obtained": self.curr_moves * 0.01,  # try to learn new moves, via menuing?

            # Make these better than nothing, but do not let these be larger than the above
            "bag_menu_action": self.rewarded_action_bag_menu * 0.0001,
            "pokemon_menu_action": self.rewared_pokemon_action * 0.0001,

            # Cut-related. Revisit later.
            "cut_coords": sum(self.cut_coords.values()) * 1.0,
            "cut_tiles": len(self.cut_tiles) * 1.0,
        }

    def _update_event_obs(self):
        for i, addr in enumerate(range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)):
            # NOTE: event obs is NOT binary, so we may miss some information there
            # self.event_obs[i] = self.bit_count(self.read_m(addr))
            val = self.read_m(addr)
            if val > 0:
                if addr not in self.experienced_events:
                    self.experienced_events.add(addr)
                    # Update event count, which is kept for the whole training run
                    if addr not in self.event_count:
                        self.event_count[addr] = 1  # unit: episode
                    else:
                        self.event_count[addr] += 1

                # NOTE: progress is very sensitive to this
                # Explore (the first run) vs. Exploit (learned path) trade-off is happening
                # Exploit: After the first run, agents know "where" to collect reward.
                #          Larger reward should come from major milestones and making progress (collect new tile scores)
                #          Having large event reward slow the agents down because it will hit every event
                #           - No discount makes story progression slower after each reset
                #           - 0-ing all known events make story progression very fast after reset, but forget events
                discount_factor = 1.0 / self.event_count[addr]
                self.event_reward[i] = self.bit_count(val) * discount_factor

        # NOTE: base_event_reward is different after reset. What's going on?
        if self.base_event_reward == 0:
            self.base_event_reward = self.event_reward.sum()

    def _update_event_reward_vars(self):
        self._update_event_obs()

        cur_rew = max(self.event_reward.sum() - self.base_event_reward - int(self.read_bit(*MUSEUM_TICKET)), 0)
        self.max_event_rew = max(cur_rew, self.max_event_rew)

        # Check KEY events
        self.badges = self.get_badges()
        self.bill_said_use_cell_separator = self.read_bit(0xD7F2, 6)
        self.got_hm01 = self.read_bit(0xD803, 0)

        # Check learn moves with item -- could be spammed later, but it's fine for now
        new_moves = self.moves_obtained.sum()
        self.just_learned_item_move = 0
        if self.read_m(0xD31D) < self.curr_item_num:  # item consumed/tossed?
            self.consumed_item_count += 1
            if new_moves > self.curr_moves:  # move learned
                self.moves_learned_with_item += 1
                self.just_learned_item_move = 1
        self.curr_moves = new_moves
        self.curr_item_num = self.read_m(0xD31D)

    @property
    def key_events_reward(self):
        return sum(
            [
                self.bill_said_use_cell_separator,
                self.got_hm01,
                self.taught_cut,
            ]
        )

    def _update_tile_reward_vars(self):
        key = self.get_game_coords()
        if key not in self.seen_tiles:
            rew = self.seen_tiles[key] = 1
        else:
            rew = 1 - self.seen_tiles[key]  # discounted
            self.seen_tiles[key] = 1
        self.tile_reward += rew

    def get_levels_reward(self, level_cap=15):
        party_levels = [
            x for x in [self.read_m(addr) for addr in PARTY_LEVEL_ADDRS[:self.party_size]] if x > 0
        ]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))

        if self.max_level_sum < level_cap:
            return self.max_level_sum
        else:
            return level_cap + (self.max_level_sum - level_cap) / 4




    ##########################################################################

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

    def step_decay_seen_coords(self):
        self.seen_tiles.update(
            (k, max(0.15, v * self.decay_factor_coords))
            for k, v in self.seen_tiles.items()
        )
        self.seen_npcs.update(
            (k, max(0.15, v * self.decay_factor_npcs))
            for k, v in self.seen_npcs.items()
        )

        # NOTE: potentially useful?
        # self.seen_map_ids *= self.step_forgetting_factor["map_ids"]
        # self.explore_map *= self.step_forgetting_factor["explore"]
        # self.explore_map[self.explore_map > 0] = np.clip(
        #     self.explore_map[self.explore_map > 0], 0.15, 1
        # )

    ##########################################################################
    # Scripting helpers below

    # def set_cursor_to_item(self, target_id=0xC4):  # 0xC4: HM cut
    #     first_item = 0xD31E
    #     for idx, offset in enumerate(range(0, 40, 2)):  # 20 items max?
    #         item_id = self.read_m(first_item + offset)
    #         if item_id == target_id:
    #             # overwrite the cursor location (wListScrollOffset)
    #             self.pyboy.set_memory_value(0xCC36, idx)
    #             return
