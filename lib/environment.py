import json
import os
import random
import uuid
from collections import deque
from pathlib import Path
from typing import Optional
import functools
from pathlib import Path

import mediapy as media
import numpy as np
from skimage.transform import resize
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from lib.global_map import GLOBAL_MAP_SHAPE, local_to_global

EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
PARTY_SIZE = 0xD163
PARTY_LEVEL_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

CUT_SEQ = [
    ((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)),
    ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),
]

CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])


# TODO: Make global map usage a configuration parameter
class RedGymEnv(Env):
    def __init__(self, env_config):
        self.video_path = Path(
            os.path.join(env_config.video_dir, env_config.session_path))
        self.save_final_state = env_config.save_final_state
        self.print_rewards = env_config.print_rewards
        self.headless = env_config.headless
        self.init_state = os.path.join(env_config.state_dir,
                                       f"{env_config.init_state}.state")
        self.act_freq = env_config.action_freq
        self.max_steps = env_config.max_steps
        self.save_video = env_config.save_video
        self.fast_video = env_config.fast_video
        self.frame_stacks = env_config.frame_stacks
        self.perfect_ivs = env_config.perfect_ivs

        # Obs space-related. TODO: avoid hardcoding?
        self.policy_obs_type = env_config.policy_obs_type
        self.output_shape = (72, 80, self.frame_stacks * 3)  # CnnLstmPolicy
        # self.output_shape = (77, 80, self.frame_stacks)
        # self.screen_output_shape = (144, 160, self.frame_stacks)
        # self.output_shape = (144, 160, self.frame_stacks * 3)
        self.coords_pad = 12
        self.enc_freqs = 8

        # NOTE: Used for saving video
        self.instance_id = str(uuid.uuid4())[:8]

        # NOTE: Used for saving video
        Path(env_config.video_dir).mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses. CHECK ME: Are these used?
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.essential_map_locations = {
            v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        }

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open(os.path.join(os.path.dirname(__file__), "events.json")) as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.pyboy, self.screen = self._setup_pyboy(env_config.gb_path)

        self.first = True

    def _setup_pyboy(self, gb_path):
        head = "headless" if self.headless else "SDL2"
        pyboy = PyBoy(
            gb_path,
            debugging=False,
            disable_input=False,
            window_type=head,
        )
        screen = pyboy.botsupport_manager().screen()
        if not self.headless:
            pyboy.set_emulation_speed(6)
        return pyboy, screen

    @functools.cached_property
    def observation_space(self):
        if self.policy_obs_type == "MultiInputPolicy":
            return spaces.Dict(
                {
                    "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                    "health": spaces.Box(low=0, high=1),
                    "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                    "badges": spaces.MultiBinary(8),
                    # "events": spaces.MultiBinary((EVENT_FLAGS_END - EVENT_FLAGS_START) * 8),
                    "map": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                        dtype=np.uint8,
                    ),
                    "recent_actions": spaces.MultiDiscrete(
                        [len(self.valid_actions)] * self.frame_stacks
                    ),
                    "seen_pokemon": spaces.MultiBinary(152),
                    "caught_pokemon": spaces.MultiBinary(152),
                    "moves_obtained": spaces.MultiBinary(0xA5),
                }
            )
        elif self.policy_obs_type in ["CnnPolicy", "MlpLstmPolicy", "CnnLstmPolicy"]:
            return spaces.Box(
                low=0, high=255, shape=self.output_shape, dtype=np.uint8
            )

    @functools.cached_property
    def action_space(self):
        return spaces.Discrete(len(self.valid_actions))

    def reset(self, seed: Optional[int] = None):
        # restart game, skipping credits
        self.explore_map_dim = 384
        if self.first:
            self.recent_screens = deque()  # np.zeros(self.output_shape, dtype=np.uint8)
            self.recent_actions = deque()  # np.zeros((self.frame_stacks,), dtype=np.uint8)
            self.seen_pokemon = np.zeros(152, dtype=np.uint8)
            self.caught_pokemon = np.zeros(152, dtype=np.uint8)
            self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
            self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
            self.init_map_mem()
            self.init_npc_mem()
            self.init_hidden_obj_mem()
            self.init_cut_mem()

        if self.first:  # or random.uniform(0, 1) < 0.5:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
            self.recent_screens.clear()
            self.recent_actions.clear()
            self.seen_pokemon.fill(0)
            self.caught_pokemon.fill(0)
            self.moves_obtained.fill(0)

            # lazy random seed setting
            if not seed:
                seed = random.randint(0, 4096)
            for _ in range(seed):
                self.pyboy.tick()

            self.explore_map *= 0
            self.init_map_mem()
            self.init_npc_mem()
            self.init_hidden_obj_mem()
            self.init_cut_mem()

        self.taught_cut = self.check_if_party_has_cut()

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0

        self.last_health = 1
        self.total_heal_health = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        self.current_event_flags_set = {}

        self.action_hist = np.zeros(len(self.valid_actions))

        # experiment!
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

        self.reset_count += 1
        self.first = False
        return self._get_obs(), {}

    def init_map_mem(self):
        # Maybe I should preallocate a giant matrix for all map ids
        # All map ids have the same size, right?
        self.seen_coords = {}
        # self.seen_global_coords = np.zeros(GLOBAL_MAP_SHAPE)
        self.seen_map_ids = np.zeros(256)

    def init_npc_mem(self):
        self.seen_npcs = {}

    def init_hidden_obj_mem(self):
        self.seen_hidden_objs = {}

    def init_cut_mem(self):
        self.cut_coords = {}
        self.cut_state = deque(maxlen=3)

    def render(self, reduce_res=True):
        # (144, 160, 3)
        game_pixels_render = self.screen.screen_ndarray()[:, :, 0:1]
        # place an overlay on top of the screen greying out places we haven't visited
        # first get our location
        player_x, player_y, map_n = self.get_game_coords()

        """
        map_height = self.read_m(0xD524)
        map_width = self.read_m(0xD525)
        print(
            self.read_m(0xC6EF),
            self.read_m(0xD524),
            self.read_m(0xD525),
            player_y,
            player_x,
        """

        # player is centered at 68, 72 in pixel units
        # 68 -> player y, 72 -> player x
        # guess we want to attempt to map the pixels to player units or vice versa
        # Experimentally determined magic numbers below. Beware
        visited_mask = np.zeros_like(game_pixels_render)
        """
        if self.taught_cut:
            cut_mask = np.zeros_like(game_pixels_render)
        else:
            cut_mask = np.random.randint(0, 255, game_pixels_render.shape, dtype=np.uint8)
        """
        # If not in battle, set the visited mask. There's no reason to process it when in battle
        if self.read_m(0xD057) == 0:
            for y in range(-72 // 16, 72 // 16):
                for x in range(-80 // 16, 80 // 16):
                    # y-y1 = m (x-x1)
                    # map [(0,0),(1,1)] -> [(0,.5),(1,1)] (cause we dont wnat it to be fully black)
                    # y = 1/2 x + .5
                    # current location tiles - player_y*8, player_x*8
                    visited_mask[
                        16 * y + 76 : 16 * y + 16 + 76,
                        16 * x + 80 : 16 * x + 16 + 80,
                        :,
                    ] = int(
                        255
                        * (
                            self.seen_coords.get(
                                (
                                    player_x + x + 1,
                                    player_y + y + 1,
                                    map_n,
                                ),
                                0.15,
                            )
                        )
                    )
                    """
                    if self.taught_cut:
                        cut_mask[
                            16 * y + 76 : 16 * y + 16 + 76,
                            16 * x + 80 : 16 * x + 16 + 80,
                            :,
                        ] = int(
                            255
                            * (
                                self.cut_coords.get(
                                    (
                                        player_x + x + 1,
                                        player_y + y + 1,
                                        map_n,
                                    ),
                                    0,
                                )
                            )
                        )
                        """
        """
        gr, gc = local_to_global(player_y, player_x, map_n)
        visited_mask = (
            255
            * np.repeat(
                np.repeat(self.seen_global_coords[gr - 4 : gr + 5, gc - 4 : gc + 6], 16, 0), 16, -1
            )
        ).astype(np.uint8)
        visited_mask = np.expand_dims(visited_mask, -1)
        """

        # game_pixels_render = np.concatenate([game_pixels_render, visited_mask, cut_mask], axis=-1)
        game_pixels_render = np.concatenate([game_pixels_render, visited_mask], axis=-1)

        if reduce_res:
            # game_pixels_render = (
            #     downscale_local_mean(game_pixels_render, (2, 2, 1))
            # ).astype(np.uint8)
            game_pixels_render = game_pixels_render[::2, ::2, :]
        return game_pixels_render

    def _get_obs(self):
        screen = self.render()
        screen = np.concatenate(
            [
                screen,
                np.expand_dims(
                    255 * resize(self.explore_map, screen.shape[:-1], anti_aliasing=False),
                    axis=-1,
                ).astype(np.uint8),
            ],
            axis=-1,
        )

        self.update_recent_screens(screen)

        if self.policy_obs_type == "MultiInputPolicy":
            # normalize to approx 0-1
            level_sum = 0.02 * sum(
                [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
            )

            return {
                "screens": np.array(self.recent_screens).reshape(self.output_shape),
                "health": np.array([self.read_hp_fraction()]),
                "level": self.fourier_encode(level_sum),
                "badges": np.array(
                    [int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8
                ),
                "events": np.array(self.read_event_bits(), dtype=np.int8),
                "recent_actions": np.array(self.recent_actions),
                "caught_pokemon": self.caught_pokemon,
                "seen_pokemon": self.seen_pokemon,
                "moves_obtained": self.moves_obtained,
            }
        else:
            return np.array(self.recent_screens).reshape(self.output_shape)

    def set_perfect_iv_dvs(self):
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(12):  # Number of offsets for IV/DV
                self.pyboy.set_memory_value(i + 17 + m, 0xFF)

    def check_if_party_has_cut(self) -> bool:
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(4):  # Number of offsets for IV/DV
                if self.pyboy.get_memory_value(i + 8 + m) == 15:
                    return True
        return False

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        # self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_health()
        self.update_pokedex()
        self.update_moves_obtained()
        self.party_size = self.read_m(0xD163)
        self.update_max_op_level()
        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        self.taught_cut = self.check_if_party_has_cut()

        info = {}
        # TODO: Make log frequency a configuration parameter
        if self.step_count % 20000 == 0:
            info = self.agent_stats(action)

        obs = self._get_obs()

        # create a map of all event flags set, with names where possible
        # if step_limit_reached:
        """
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        # TODO this currently seems to be broken!
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")
        """

        self.step_count += 1

        # return obs, new_reward, self.step_count > self.max_steps, False, info
        return obs, new_reward, False, False, info

    def find_neighboring_sign(self, sign_id, player_direction, player_x, player_y) -> bool:
        sign_y = self.pyboy.get_memory_value(0xD4B1 + (2 * sign_id))
        sign_x = self.pyboy.get_memory_value(0xD4B1 + (2 * sign_id + 1))

        # Check if player is facing the sign (skip sign direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        # We are making the assumption that a player will only ever be 1 space away
        # from a sign
        return (
            (player_direction == 0 and sign_x == player_x and sign_y == player_y + 1)
            or (player_direction == 4 and sign_x == player_x and sign_y == player_y - 1)
            or (player_direction == 8 and sign_y == player_y and sign_x == player_x - 1)
            or (player_direction == 0xC and sign_y == player_y and sign_x == player_x + 1)
        )

    def find_neighboring_npc(self, npc_id, player_direction, player_x, player_y) -> int:
        npc_y = self.pyboy.get_memory_value(0xC104 + (npc_id * 0x10))
        npc_x = self.pyboy.get_memory_value(0xC106 + (npc_id * 0x10))

        # Check if player is facing the NPC (skip NPC direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        if (
            (player_direction == 0 and npc_x == player_x and npc_y > player_y)
            or (player_direction == 4 and npc_x == player_x and npc_y < player_y)
            or (player_direction == 8 and npc_y == player_y and npc_x < player_x)
            or (player_direction == 0xC and npc_y == player_y and npc_x > player_x)
        ):
            # Manhattan distance
            return abs(npc_y - player_y) + abs(npc_x - player_x)

        return False

    def run_action_on_emulator(self, action):
        self.action_hist[action] += 1

        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8 and action < len(self.release_actions):
                # release button
                self.pyboy.send_input(self.release_actions[action])

            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()

        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if self.taught_cut:
            player_direction = self.pyboy.get_memory_value(0xC109)
            x, y, map_id = self.get_game_coords() # x, y, map_id
            if player_direction == 0: # down
                coords = (x, y+1, map_id)
            if player_direction == 4:
                coords = (x, y-1, map_id)
            if player_direction == 8:
                coords = (x-1, y, map_id)
            if player_direction == 0xC:
                coords = (x+1, y, map_id)
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
                self.cut_coords[coords] = 1
            elif self.cut_state == CUT_GRASS_SEQ:
                self.cut_coords[coords] = 0.3
            elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                self.cut_coords[coords] = 0.005

        # check if the font is loaded
        if self.pyboy.get_memory_value(0xCFC4):
            # check if we are talking to a hidden object:
            player_direction = self.pyboy.get_memory_value(0xC109)
            player_y_tiles = self.pyboy.get_memory_value(0xD361)
            player_x_tiles = self.pyboy.get_memory_value(0xD362)
            if (
                self.pyboy.get_memory_value(0xCD3D) != 0x0
                and self.pyboy.get_memory_value(0xCD3E) != 0x0
            ):
                # add hidden object to seen hidden objects
                self.seen_hidden_objs[
                    (
                        self.pyboy.get_memory_value(0xD35E),
                        self.pyboy.get_memory_value(0xCD3F),
                    )
                ] = 1
            elif any(
                self.find_neighboring_sign(
                    sign_id, player_direction, player_x_tiles, player_y_tiles
                )
                for sign_id in range(self.pyboy.get_memory_value(0xD4B0))
            ):
                pass
            else:
                # get information for player
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
                    for npc_id in range(1, self.pyboy.get_memory_value(0xD4E1))
                )
                npc_candidates = [x for x in npc_distances if x[0]]
                if npc_candidates:
                    _, npc_id = min(npc_candidates, key=lambda x: x[0])
                    self.seen_npcs[(self.pyboy.get_memory_value(0xD35E), npc_id)] = 1

        if self.save_video and self.fast_video:
            self.add_video_frame()

    def agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return {
            "stats": {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_location": self.get_map_location(map_n),
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord": sum(self.seen_coords.values()),  # np.sum(self.seen_global_coords),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_heal_health,
                "action_hist": self.action_hist,
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "moves_obtained": int(sum(self.moves_obtained)),
                "opponent_level": self.max_opponent_level,
                "met_bill": int(self.read_bit(0xD7F1, 0)),
                "used_cell_separator_on_bill": int(self.read_bit(0xD7F2, 3)),
                "ss_ticket": int(self.read_bit(0xD7F2, 4)),
                "met_bill_2": int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": int(self.read_bit(0xD7F2, 7)),
                "got_hm01": int(self.read_bit(0xD803, 0)),
                "rubbed_captains_back": int(self.read_bit(0xD803, 1)),
                "taught_cut": int(self.check_if_party_has_cut()),
                "cut_coords": sum(self.cut_coords.values()),
            },
            "reward": self.get_game_state_reward(),
            "reward/reward_sum": sum(self.get_game_state_reward().values()),
            "pokemon_exploration_map": self.explore_map,
        }

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.video_path
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f"full_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        model_name = Path(f"model_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(f"map_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        self.seen_coords[(x_pos, y_pos, map_n)] = 1
        self.explore_map[local_to_global(y_pos, x_pos, map_n)] = 1
        # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
        self.seen_map_ids[map_n] = 1

    def get_explore_map(self):
        explore_map = np.zeros(GLOBAL_MAP_SHAPE)
        for (x, y, map_n), v in self.seen_coords.items():
            gy, gx = local_to_global(y, x, map_n)
            if gy >= explore_map.shape[0] or gx >= explore_map.shape[1]:
                print(f"coord out of bounds! global: ({gx}, {gy}) game: ({x}, {y}, {map_n})")
            else:
                explore_map[gy, gx] = v

        return explore_map

    def update_recent_screens(self, cur_screen):
        # self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        # self.recent_screens[:, :, 0] = cur_screen[:, :, 0]
        self.recent_screens.append(cur_screen)
        if len(self.recent_screens) > self.frame_stacks:
            self.recent_screens.popleft()

    def update_recent_actions(self, action):
        # self.recent_actions = np.roll(self.recent_actions, 1)
        # self.recent_actions[0] = action
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.frame_stacks:
            self.recent_actions.popleft()

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    # Override this method for custom reward functions
    def get_game_state_reward(self, print_stats=False):
        raise NotImplementedError

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit)
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]

    def update_max_op_level(self):
        # opp_base_level = 5
        opponent_level = (
            max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]])
            # - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_health(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                self.total_heal_health += cur_health - self.last_health
            else:
                self.died_count += 1

    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.pyboy.get_memory_value(i + 0xD2F7)
            seen_mem = self.pyboy.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8 * i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8 * i + j] = 1 if seen_mem & (1 << j) else 0

    def update_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if self.pyboy.get_memory_value(i) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(i + j + 8)
                    if move_id != 0:
                        if move_id != 0:
                            self.moves_obtained[move_id] = 1
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.get_memory_value(0xDA80)):
            offset = i * box_struct_length + 0xDA96
            if self.pyboy.get_memory_value(offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1

    def read_hp_fraction(self):
        hp_sum = sum(
            [self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
        )
        max_hp_sum = sum(
            [self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]]
        )
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
        # return bits.bit_count()

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_map_location(self, map_idx):
        map_locations = {
            0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
            1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
            2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
            3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
            62: {
                "name": "Invaded house (Cerulean City)",
                "coordinates": np.array([290, 227]),
            },
            63: {
                "name": "trade house (Cerulean City)",
                "coordinates": np.array([290, 212]),
            },
            64: {
                "name": "Pokémon Center (Cerulean City)",
                "coordinates": np.array([290, 197]),
            },
            65: {
                "name": "Pokémon Gym (Cerulean City)",
                "coordinates": np.array([290, 182]),
            },
            66: {
                "name": "Bike Shop (Cerulean City)",
                "coordinates": np.array([290, 167]),
            },
            67: {
                "name": "Poké Mart (Cerulean City)",
                "coordinates": np.array([290, 152]),
            },
            35: {"name": "Route 24", "coordinates": np.array([250, 235])},
            36: {"name": "Route 25", "coordinates": np.array([270, 267])},
            12: {"name": "Route 1", "coordinates": np.array([70, 43])},
            13: {"name": "Route 2", "coordinates": np.array([70, 151])},
            14: {"name": "Route 3", "coordinates": np.array([100, 179])},
            15: {"name": "Route 4", "coordinates": np.array([150, 197])},
            33: {"name": "Route 22", "coordinates": np.array([20, 71])},
            37: {"name": "Red house first", "coordinates": np.array([61, 9])},
            38: {"name": "Red house second", "coordinates": np.array([61, 0])},
            39: {"name": "Blues house", "coordinates": np.array([91, 9])},
            40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
            41: {
                "name": "Pokémon Center (Viridian City)",
                "coordinates": np.array([100, 54]),
            },
            42: {
                "name": "Poké Mart (Viridian City)",
                "coordinates": np.array([100, 62]),
            },
            43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
            44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
            47: {
                "name": "Gate (Viridian City/Pewter City) (Route 2)",
                "coordinates": np.array([91, 143]),
            },
            49: {"name": "Gate (Route 2)", "coordinates": np.array([91, 115])},
            50: {
                "name": "Gate (Route 2/Viridian Forest) (Route 2)",
                "coordinates": np.array([91, 115]),
            },
            51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
            52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
            53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
            54: {
                "name": "Pokémon Gym (Pewter City)",
                "coordinates": np.array([49, 176]),
            },
            55: {
                "name": "House with disobedient Nidoran♂ (Pewter City)",
                "coordinates": np.array([51, 184]),
            },
            56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
            57: {
                "name": "House with two Trainers (Pewter City)",
                "coordinates": np.array([51, 184]),
            },
            58: {
                "name": "Pokémon Center (Pewter City)",
                "coordinates": np.array([45, 161]),
            },
            59: {
                "name": "Mt. Moon (Route 3 entrance)",
                "coordinates": np.array([153, 234]),
            },
            60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
            61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
            68: {
                "name": "Pokémon Center (Route 3)",
                "coordinates": np.array([135, 197]),
            },
            193: {
                "name": "Badges check gate (Route 22)",
                "coordinates": np.array([0, 87]),
            },  # TODO this coord is guessed, needs to be updated
            230: {
                "name": "Badge Man House (Cerulean City)",
                "coordinates": np.array([290, 137]),
            },
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return {
                "name": "Unknown",
                "coordinates": np.array([80, 0]),
            }  # TODO once all maps are added this case won't be needed