import pufferlib.models


class RecurrentWrapper(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

# NOTE: see RedGymEnv.screen_output_shape = (72, 80, 3 * self.frame_stacks)
class ConvolutionalPolicy(pufferlib.models.Convolutional):
    def __init__(
        self, env,
        input_size=512, hidden_size=512, output_size=512, framestack=3, flat_size=1920
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