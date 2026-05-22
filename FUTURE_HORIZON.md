# Action decoder future horizon (z0_future)

During inference the action decoder needs **noise-free future frames** — referred to as `z0_future` in the diffusion model sense (the clean, σ=0 sample) — to condition the Cosmos-Predict2 backbone before extracting hidden states for cross-attention.

## How the total sequence length is set

The backbone always processes a fixed-length video of **61 frames** (set via `num_frames=61` in [`configs/defaults/data_video.py`](./model/cosmos_predict2/configs/defaults/data_video.py)).

## Splitting past vs. future

The split between conditioning (observed) frames `T` and future frames is enforced at call time:

```python
# video2world2action_gtvid.py:46-48
B, _C, T, _H, _W = input_vid.shape
assert T in {1, 5}
assert gt_future_vid.shape[2] == 61 - T
```

`gt_future_vid` (the noise-free z0_future) therefore has **61 − T** video frames:

| Conditioning frames T | z0_future video frames (5 Hz) |
|-----------------------|-------------------------------|
| 5 (standard)          | **56**                        |
| 1                     | **60**                        |

The standard setting is T = 5, matching `obs/workspace_rgb: horizon: 5` in [`configs/dataloading/policy_io/libero.yaml`](./model/cosmos_predict2/configs/dataloading/policy_io/libero.yaml).

## Action decoder horizon (lowdim actions)

For the lowdim action head the future horizon is derived from `max_horizon` (the action decoder DiT's sequence capacity) minus `HO` (the number of observed lowdim state steps):

```python
# world2action.py:207-208
T = self.config.net.max_horizon
HA = T - HO   # HO = 1 (single lowdim obs step)
```

`max_horizon` is set per dataset in [`configs/defaults/world2action_pipe.py`](./model/cosmos_predict2/configs/defaults/world2action_pipe.py):

| Dataset | `max_horizon` | HO | Action horizon HA (20 Hz) |
|---------|---------------|----|---------------------------|
| Libero  | 61            | 1  | **60 steps ≈ 3 s**        |
| Bridge  | 16            | 1  | **15 steps ≈ 0.75 s**     |

The lowdim action fields (`eef_pos_ref_delta_lowdim`, `eef_rot_ref_delta_lowdim`, `gripper_action_lowdim`) all use `horizon: 60` for Libero and `horizon: 15` for Bridge, confirming this — see the `policy_io` YAML files in [`configs/dataloading/policy_io/`](./model/cosmos_predict2/configs/dataloading/policy_io/).
