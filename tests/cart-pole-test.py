import torch
import tqdm
import gym
import multiprocessing
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    ExplorationType,
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardScaling,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyModule, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

import spatialNN
from spatialNN import Model

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

env = TransformedEnv(
    GymEnv("CartPole-v1", device = device), 
    Compose(
        InitTracker(),
        StepCounter()
        )
    )

qval = QValueModule(spec=env.action_spec)

spatial = Mod(
    spatialNN.Model.Spatial_Model(
        n_dimensions = 2, 
        n_neurons = 60, 
        inputs = 4, 
        outputs = 2,
    ),
    in_keys=["observation"],
    out_keys=["action_value"]
)

spatial_recurrent = Mod(
    spatialNN.Model.Spatial_Model(
        n_dimensions = 2, 
        n_neurons = 60, 
        inputs = 4, 
        outputs = 2,
        recurrent_mode = True
    ),
    in_keys=["observation"],
    out_keys=["action_value"]
)

stoch_policy = Seq(spatial, qval)

exploration_module = EGreedyModule(
    annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
)

stoch_policy = Seq(
    stoch_policy,
    exploration_module,
)
print("Initial stoch_policy value:", stoch_policy(env.reset()))

policy = Seq(spatial_recurrent, qval)
print("Initial policy value:", policy(torch.unsqueeze(env.reset(), 0)))



loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)

updater = SoftUpdate(loss_fn, eps=0.95)

optim = torch.optim.Adam(policy.parameters(), lr=3e-1)

collector = SyncDataCollector(env, stoch_policy, frames_per_batch=50, total_frames=100, device=device)
rb = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(20_000), batch_size=1, prefetch=10
)

utd = 16
pbar = tqdm.tqdm(total=1_000_000)
longest = 0

traj_lens = []
for i, data in enumerate(collector):
    if i == 0:
        print(
            "Let us print the first batch of data.\nPay attention to the key names "
            "which will reflect what can be found in this data structure, in particular: "
            "the output of the QValueModule (action_values, action and chosen_action_value),"
            "the 'is_init' key that will tell us if a step is initial or not, and the "
            "recurrent_state keys.\n",
            data,
        )
    pbar.update(data.numel())
    # it is important to pass data that is not flattened
    rb.extend(data.unsqueeze(0).to_tensordict().cpu())
    for _ in range(utd):
        s = rb.sample().to(device, non_blocking=True)
        loss_vals = loss_fn(s)
        loss_vals["loss"].backward()
        optim.step()
        #for param in policy.parameters():
        #    print(param)
        optim.zero_grad()
    longest = max(longest, data["step_count"].max().item())
    pbar.set_description(
        f"steps: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}, action_spread: {data['action'].sum(0)}"
    )
    exploration_module.step(data.numel())
    updater.step()

    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        rollout = env.rollout(10000, stoch_policy)
        traj_lens.append(rollout.get(("next", "step_count")).max().item())


if traj_lens:
    from matplotlib import pyplot as plt

    plt.plot(traj_lens)
    plt.xlabel("Test collection")
    plt.title("Test trajectory lengths")

    plt.savefig("Traj_lens.png")