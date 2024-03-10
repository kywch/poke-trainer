import torch
from torch import nn

import pufferlib.models


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

# NOTE: This is CnnLstmPolicy with Recurrent wrapper
class Policy(pufferlib.models.Convolutional):
    def __init__(
        self, env,
        input_size=512, hidden_size=512, output_size=512, framestack=3, flat_size=1920
        # 14336 for self.output_shape = (144, 160, self.frame_stacks * 3)
    ):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
            channels_last=True,
        )


# CHECK ME: Is this used?
class MultiConvolutionPolicy(pufferlib.models.Policy):
    def __init__(
        self,
        env,
        screen_framestack: int,
        mask_framestack: int,
        global_map_frame_stack: int,
        screen_flat_size: int,
        mask_flat_size: int,
        global_map_flat_size: int,
        hidden_size=512,
        output_size=512,
        channels_last: bool = False,
        downsample: int = 1,
    ):
        super().__init__(env)
        self.num_actions = self.action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.screen_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(screen_framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(screen_flat_size, hidden_size)),
            nn.ReLU(),
        )

        self.masks_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(mask_framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(mask_flat_size, hidden_size)),
            nn.ReLU(),
        )

        self.global_map_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(global_map_frame_stack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(global_map_flat_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, self.num_actions), std=0.01
        )
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def encode_observations(self, observations):
        output = []
        for okey, network, scalefactor in zip(
            ("screen", "masks", "global_map"),
            (self.screen_network, self.masks_network, self.global_map_network),
            (255.0, 255.0, 255.0),
        ):
            observation = observations[okey]
            if self.channels_last:
                observations = observation.permute(0, 3, 1, 2)
            if self.downsample > 1:
                observation = observation[:, :, :: self.downsample, :: self.downsample]
            output.append(network(observation.float() / scalefactor))
        return torch.cat(output), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value




# class Policy(pufferlib.models.ProcgenResnet):
#     def __init__(self, env, cnn_width=16, mlp_width=512):
#         super().__init__(
#             env=env,
#             cnn_width=cnn_width,
#             mlp_width=mlp_width,
#         )

'''
class Policy(pufferlib.models.Policy):
    def __init__(
        self,
        env,
        *args,
        framestack,
        flat_size,
        input_size=512,
        hidden_size=512,
        output_size=512,
        channels_last=True,
        downsample=1,
        **kwargs,
    ):
        """The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer

        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without."""
        super().__init__(env)
        self.num_actions = self.action_space.n
        self.channels_last = channels_last
        self.downsample = downsample

        self.network = torch.nn.Sequential(
            # A resnet 18
            # a resnet 34 would be [3, 4, 6, 3]
            ResNet(BasicBlock, [2, 2, 2, 2], input_channels=2, num_classes=1000),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(output_size),
            torch.nn.ReLU()
        )

        self.actor = pufferlib.pytorch.layer_init(
            torch.nn.Linear(output_size, self.num_actions), std=0.01
        )
        self.value_fn = pufferlib.pytorch.layer_init(torch.nn.Linear(output_size, 1), std=1)

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, :: self.downsample, :: self.downsample]
        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
'''