from typing import Optional
from collections import deque

import numpy as np
from gymnasium import spaces
import pufferlib

from pokemonred_puffer.environment import (
    RedGymEnv,
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    PARTY_LEVEL_ADDRS,
    CUT_SEQ,
    CUT_GRASS_SEQ,
    CUT_FAIL_SEQ,
)

ITEM_COUNT = 0xD31D
BATTLE_FLAG = 0xD057
TEXT_BOX_UP = 0xCFC4
MUSEUM_TICKET = (0xD754, 0)

GYM_3_TREE = (3984, 4512, 5)

PRESS_BUTTON_A = 5

BASE_ENEMY_LEVEL = 4

# Map ids to visit in sequene -- CANNOT use the map id more than once
STORY_PROGRESS = [40, 0, 12, 1,     # Oaks lab - Pallet town - Route 1 - Veridian city
                  13, 51, 2, 54,    # Route 2 - Viridian forest - Pewter city - Pewter gym
                  14, 59, 60, 61,   # Route 3 - Mt Moon: Route 3 - B1F - B2F
                  15, 3, 65, 35,    # Route 4 - Cerulean city - Cerulean gym - Route 24
                  36, 16, 17, 5,    # Route 25 - Route 5 - Route 6 - Vermilion city
                  92]               # Vermilion gym (can go there after learning cut)


class CustomRewardEnv(RedGymEnv):
    def __init__(self, env_config: pufferlib.namespace, reward_config: pufferlib.namespace):
        super().__init__(env_config)
        self.default_max_steps = env_config.max_steps
        self.essential_map_locations = {
            v: i for i, v in enumerate(STORY_PROGRESS)
        }

        self.event_count = {}  # kept for whole training run, take this when checkpointing
        self.experienced_events = set()  # reset every episode
        self.event_reward = np.zeros(320)
        self.base_event_reward = 0

        # NOTE: decaying seen coords/tiles makes reward dense, making the place more "sticky"
        self.tile_reward = 0
        self.seen_tiles = {}
        self.npc_reward = 0
        self.talked_npcs = {}

        self.cut_success = set()
        self.cut_attempt = 0
        self.cut_close = {}

        self._reset_reward_vars()

        # Reward weights
        self.decay_frequency = reward_config["decay_frequency"]
        self.decay_factor_coords = reward_config["decay_factor_coords"]
        self.decay_factor_npcs = reward_config["decay_factor_npcs"]
        self.decay_floor = reward_config["decay_floor"]

        self.weight_tile_exploration = reward_config["weight_tile_exploration"]
        self.weight_seen_npcs = reward_config["weight_seen_npcs"]

        self.weight_badge = reward_config["weight_badge"]
        self.weight_key_event = reward_config["weight_key_event"]
        self.weight_map_progress = reward_config["weight_map_progress"]
        self.weight_max_opponent_level = reward_config["weight_max_opponent_level"]

        self.weight_party_size = reward_config["weight_party_size"]
        self.weight_level_sum = reward_config["weight_level_sum"]
        self.weight_level_cap = reward_config["weight_level_cap"]
        self.weight_over_level_multiplier = reward_config["weight_over_level_multiplier"]

        self.weight_learned_moves_with_item = reward_config["weight_learned_moves_with_item"]
        self.weight_learned_moves = reward_config["weight_learned_moves"]

        self.weight_seen_pokemon = reward_config["weight_seen_pokemon"]
        self.weight_max_event = reward_config["weight_max_event"]
        self.weight_hidden_obj = reward_config["weight_hidden_obj"]

        self.weight_menu_reward = reward_config["weight_menu_reward"]
        self.cooldown_menu_normal = reward_config["cooldown_menu_normal"]
        self.cooldown_menu_boost = reward_config["cooldown_menu_boost"]

        # revisit later
        self.weight_cut_attempt = reward_config["weight_cut_attempt"]
        self.weight_cut_success = reward_config["weight_cut_success"]
        self.weight_cut_close = reward_config["weight_cut_close"]
        self.cooldown_cut = reward_config["cooldown_cut"]

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
                "boost_menu_reward": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
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
            "map_progress": np.array(self.max_map_progress + 1, dtype=np.uint8),  # using 32 one-hot, 21 map locs
            "num_badge": np.array(self.get_badges(), dtype=np.uint8),  # using 9 one-hot
            "party_size": np.array(self.party_size, dtype=np.uint8),  # using 8 one-hot -- CHECK ME: below 8?
            "seen_pokemon": np.array(self.seen_pokemon.sum(), dtype=np.uint8),  # a great proxy for game progression
            #"direction": np.array(self.pyboy.get_memory_value(0xC109) // 4, dtype=np.uint8),
            "boost_menu_reward": np.array(self.boost_menu_reward, dtype=np.uint8),
            "cut_in_party": np.array(self.taught_cut, dtype=np.uint8),
        }

    def reset(self, seed: Optional[int] = None):
        # Load the state and reset all the RedGymEnv vars
        obs, info = super().reset(seed)

        # Treat the initial episode as a short tutorial
        #self.max_steps = self.default_max_steps / 2 if self.first is True else self.default_max_steps

        # NOTE: these dict are used for exploration decay within episode
        # So they should be reset every episode
        self.seen_tiles.clear()
        self.talked_npcs.clear()
        if self.first is False:
            self.seen_coords.clear()
            self.seen_npcs.clear()
            self.seen_hidden_objs.clear()
            self.cut_coords.clear()

        self.total_reward = 0  # NOTE: super.reset() updates this before resetting reward vars
        self._reset_reward_vars()

        return obs, info

    def _reset_reward_vars(self):
        self.boost_menu_reward = False

        #self.event_obs.fill(0)
        self.experienced_events.clear()
        self.event_reward.fill(0)
        self.base_event_reward = 0
        self.max_event_rew = 0
        self._update_event_obs()
        self.consumed_item_count = 0

        self.max_level_sum = 0
        self.tile_reward = 0
        self.npc_reward = 0

        # KEY events
        self.badges = 0
        self.key_events_to_cut = []

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

        # Track cut tiles
        self.cut_success.clear()
        self.cut_close.clear()
        self.cut_attempt = 0
        self.cut_reward_cooldown = 0

    def step(self, action):
        if self.menu_reward_cooldown > 0:
            self.menu_reward_cooldown -= 1
        if self.cut_reward_cooldown > 0:
            self.cut_reward_cooldown -= 1

        self.boost_menu_reward = self.got_hm01_cut_but_not_learned_yet()
        # if self.boost_menu_reward:
        #     # NOTE: only for HM cut now
        #     self.set_cursor_to_item(target_id=0xC4)  # 0xC4: HM cut
        #     pass

        obs, rew, reset, _, info = super().step(action)

        self._update_menu_reward_vars(action)

        # NOTE: info is not always provided
        if "stats" in info:
            info["stats"]["boost_menu_reward"] = self.boost_menu_reward
            info["stats"]["learn_with_item"] = self.moves_learned_with_item
            info["stats"]["action_bag_menu_count"] = self.action_bag_menu_count
            info["stats"]["rewarded_action_bag_menu"] = self.rewarded_action_bag_menu
            info["stats"]["pokemon_action_count"] = self.pokemon_action_count
            info["stats"]["rewared_pokemon_action"] = self.rewared_pokemon_action
            info["stats"]["consumed_item_count"] = self.consumed_item_count

            # Decay-applied rewards
            info["stats"]["new_event_reward"] = self.event_reward.sum()
            info["stats"]["new_tile_reward"] = self.tile_reward
            info["stats"]["new_npc_reward"] = self.npc_reward

            # Cut-related infos
            info["stats"]["cut_attempt"] = self.cut_attempt
            info["stats"]["cut_success"] = len(self.cut_success)
            info["stats"]["cut_close_cnt"] = len(self.cut_close)

        return obs, rew, reset, False, info

    def _update_menu_reward_vars(self, action):
        # Check menu action
        if action is not None and action == PRESS_BUTTON_A:
            if self.check_if_in_bag_menu():
                self.action_bag_menu_count += 1
                if self.menu_reward_cooldown == 0:
                    self.rewarded_action_bag_menu += 1
                    self.menu_reward_cooldown = self.cooldown_menu_normal
            if self.check_if_in_pokemon_menu():
                self.pokemon_action_count += 1
                if self.menu_reward_cooldown == 0:
                    self.rewared_pokemon_action += 1
                    self.menu_reward_cooldown = self.cooldown_menu_normal

    # Reward is computed with update_reward(), which calls get_game_state_reward()
    def update_reward(self):
        if self.boost_menu_reward is True:
            # NOTE: this is extreme item-action reward boosting
            return self._process_menu_boost_reward()

        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def got_hm01_cut_but_not_learned_yet(self):
        return not self.taught_cut and all(self.key_events_to_cut)

    def _process_menu_boost_reward(self):
        # Use/toss all the items -- hm 01 cannot be tossed
        self._update_learned_moves_with_item_vars()
        if self.just_learned_item_move > 0:
            return 1.0
        if self.just_consumed_item > 0:
            return 0.1

        # Encourage going to action bag menu with small reward
        if self.seen_action_bag_menu == 1 and self.menu_reward_cooldown == 0:
            self.menu_reward_cooldown = self.cooldown_menu_boost
            self.action_bag_menu_count += 1
            self.rewarded_action_bag_menu += 1
            return 0.0001

        # other actions -- no reward
        return 0

    # TODO: make the reward weights configurable
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        if self.step_count % self.decay_frequency == 0:
            self.step_decay_memory_tile_npc()

        self._update_event_reward_vars()
        self._update_learned_moves_with_item_vars()
        self._update_tile_reward_vars()
        self._update_npc_reward_vars()
        self._check_cut()

        return {
            # Main milestones for story progression
            "badge": self.badges * self.weight_badge,
            "taught_cut": self.taught_cut * self.weight_badge,
            "key_events": sum(self.key_events_to_cut) * self.weight_key_event,
            "map_progress": self.max_map_progress * self.weight_map_progress,
            "opponent_level": (self.max_opponent_level - BASE_ENEMY_LEVEL) * self.weight_max_opponent_level,

            # Party strength proxy
            "party_size": self.party_size * self.weight_party_size,
            "level": self.level_reward * self.weight_level_sum,

            # Important skill: learning moves with items
            "learn_with_item": self.moves_learned_with_item * self.weight_learned_moves_with_item,
            "moves_obtained": self.curr_moves * self.weight_learned_moves,

            # Exploration: bias agents' actions with weight for each new gain
            # These kick in when agent is "stuck"

            # NOTE: exploring "newer" tiles is the main driver of progression
            # Visit decay makes the explore reward "dense" ... little reward everywhere
            # so agents are motivated to explore new coords and/or revisit old coords
            "explore_tile": self.tile_reward * self.weight_tile_exploration,

            # First, always search for new pokemon and events
            "seen_pokemon": self.seen_pokemon.sum() * self.weight_seen_pokemon,  # more related to story progression?

            # NOTE: there seems to be a lot of irrevant events?
            # event weight ~0: after 1st reset, agents go straight to the next target, but after 2-3, it forgets to make progress
            # event weight 1: atter 1st reset, agents stick to "old" events, that guarantee reward ... so does not make progress
            # after seeing this, implemented the experienced event reward discounting
            "event": self.max_event_rew * self.weight_max_event,

            # If the above doesn't work, try these in the order of importance
            "explore_hidden_objs": len(self.seen_hidden_objs) * self.weight_hidden_obj,
            "explore_npcs": self.npc_reward * self.weight_seen_npcs,  # talk to npcs, getting discounted rew for revisiting

            # Make these better than nothing, but do not let these be larger than the above
            "bag_menu_action": self.rewarded_action_bag_menu * self.weight_menu_reward,
            "pokemon_menu_action": self.rewared_pokemon_action * self.weight_menu_reward,

            # Charge cost per step, to encourage any action
            #"step_cost": self.step_count * -0.0003,  # ~40 at 132k steps

            # Cut-related. Revisit later.
            "cut_success":  len(self.cut_success) * self.weight_cut_success,
            "cut_attempt": self.cut_attempt * self.weight_cut_attempt,
            "cut_close": sum(self.cut_close.values()),
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

                #discount_factor = 1.0 / self.event_count[addr]
                #discount_factor = (11 - self.event_count[addr]) * 0.1 if self.event_count[addr] < 5 else 0.7
                self.event_reward[i] = self.bit_count(val) # * discount_factor

        # NOTE: base_event_reward is different after reset. What's going on?
        if self.base_event_reward == 0:
            self.base_event_reward = self.event_reward.sum()

    def _update_event_reward_vars(self):
        self._update_event_obs()

        cur_rew = max(self.event_reward.sum() - self.base_event_reward - int(self.read_bit(*MUSEUM_TICKET)), 0)
        self.max_event_rew = max(cur_rew, self.max_event_rew)

        # Check KEY events
        self.badges = self.get_badges()
        rival_3 = self.read_m(0xD665) == 4  # beat rival before the captain
        self.key_events_to_cut = [
            self.read_bit(0xD7F1, 0),  # met bill
            self.read_bit(0xD7F2, 3),  # used cell separator on bill
            self.read_bit(0xD7F2, 4),  # ss ticket
            self.read_bit(0xD7F2, 5),  # met bill 2
            self.read_bit(0xD7F2, 6),  # bill said use cell separator
            self.read_bit(0xD7F2, 7),  # left bills house after helping
            rival_3,                   # beat rival before the captain
            self.read_bit(0xD803, 1),  # rubbed_captain
            self.read_bit(0xD803, 0),  # got hm 01
        ]

    def _update_learned_moves_with_item_vars(self):
        # Check learn moves with item -- could be spammed later, but it's fine for now
        new_moves = self.moves_obtained.sum()
        self.just_consumed_item = 0
        self.just_learned_item_move = 0
        if self.read_m(ITEM_COUNT) < self.curr_item_num:  # item consumed/tossed?
            self.consumed_item_count += 1
            self.just_consumed_item = 1
            if new_moves > self.curr_moves:  # move learned
                self.moves_learned_with_item += 1
                self.just_learned_item_move = 1
        self.curr_moves = new_moves
        self.curr_item_num = self.read_m(ITEM_COUNT)

    # @property
    # def key_events_reward(self):
    #     return sum(self.key_events_to_cut)

    @property
    def level_reward(self):
        party_levels = [
            x for x in [self.read_m(addr) for addr in PARTY_LEVEL_ADDRS[:self.party_size]] if x > 0
        ]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))

        level_cap = self.weight_level_cap
        weight_over_cap = self.weight_over_level_multiplier
        if self.max_level_sum < level_cap:
            return self.max_level_sum
        else:
            return level_cap + (self.max_level_sum - level_cap) * weight_over_cap

    def _update_tile_reward_vars(self):
        key = self.get_game_coords()
        if key not in self.seen_tiles:
            rew = self.seen_tiles[key] = 1
        else:
            rew = 1 - self.seen_tiles[key]
            self.seen_tiles[key] = 1

        self.tile_reward += rew

    # NOTE: this is duplicate from pokemonred_puffer. TODO: remove redunduncy
    def _update_npc_reward_vars(self):
        # if self.step_count % self.forget_frequency_npc == 0:
        #     self.talked_npcs.clear()

        if self.read_m(BATTLE_FLAG) == 0 and self.pyboy.get_memory_value(TEXT_BOX_UP) > 0:
            player_direction = self.pyboy.get_memory_value(0xC109)
            player_y = self.pyboy.get_memory_value(0xC104)
            player_x = self.pyboy.get_memory_value(0xC106)
            # get the npc who is closest to the player and facing them
            # we go through all npcs because there are npcs like
            # nurse joy who can be across a desk and still talk to you

            # npc_id 0 is the player
            npc_distances = (
                (
                    self.find_neighboring_npc(npc_id, player_direction, player_x, player_y),
                    npc_id,
                )
                for npc_id in range(1, self.pyboy.get_memory_value(0xD4E1))  # WNUMSPRITES
            )
            npc_candidates = [x for x in npc_distances if x[0]]

            # interacted with an npc
            if npc_candidates:
                map_id = self.pyboy.get_memory_value(0xD35E)
                _, npc_id = min(npc_candidates, key=lambda x: x[0])

                # Apply npc memory decay
                if (map_id, npc_id) not in self.talked_npcs:
                    rew = self.talked_npcs[(map_id, npc_id)] = 1
                else:
                    rew = 1 - self.talked_npcs[(map_id, npc_id)]
                    self.talked_npcs[(map_id, npc_id)] = 1
                self.npc_reward += rew

    def step_decay_memory_tile_npc(self):
        self.seen_tiles.update(
            (k, max(self.decay_floor, v * self.decay_factor_coords))
            for k, v in self.seen_tiles.items()
        )
        self.talked_npcs.update(
            (k, max(self.decay_floor, v * self.decay_factor_npcs))
            for k, v in self.talked_npcs.items()
        )

        # NOTE: potentially useful?
        # self.seen_map_ids *= self.step_forgetting_factor["map_ids"]
        # self.explore_map *= self.step_forgetting_factor["explore"]
        # self.explore_map[self.explore_map > 0] = np.clip(
        #     self.explore_map[self.explore_map > 0], 0.15, 1
        # )

    # NOTE: this is from thatguy env
    def _check_cut(self):
        if self.taught_cut > 0 and self.read_m(BATTLE_FLAG) == 0:
            player_direction = self.pyboy.get_memory_value(0xC109)
            x, y, map_id = self.get_game_coords()  # x, y, map_id
            if player_direction == 0:  # down
                coords = (x, y + 1, map_id)
            if player_direction == 4:
                coords = (x, y - 1, map_id)
            if player_direction == 8:
                coords = (x - 1, y, map_id)
            if player_direction == 0xC:
                coords = (x + 1, y, map_id)
            self.cut_state.append(
                (
                    self.pyboy.get_memory_value(0xCFC6),
                    self.pyboy.get_memory_value(0xCFCB),
                    self.pyboy.get_memory_value(0xCD6A),
                    self.pyboy.get_memory_value(0xD367),
                    self.pyboy.get_memory_value(0xD125),
                    self.pyboy.get_memory_value(0xCD3D),
                )
            )

            if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                self.cut_success.add(coords)

            elif self.cut_state == CUT_GRASS_SEQ or \
                 deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:

                # distance from the target tree
                # vermilion city gym: (3984, 4512, 5)
                # If the cut was attempted near the target tree, give reward
                if map_id == 5 and GYM_3_TREE not in self.cut_success:
                    dist_to_target = max(abs(x - 3984), abs(y - 4512))
                    if dist_to_target < 6:
                        self.cut_close[(coords, player_direction)] = (6 - dist_to_target) * self.weight_cut_close

                elif self.cut_reward_cooldown == 0:
                    self.cut_attempt += 1
                    self.cut_reward_cooldown = self.cooldown_cut



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
