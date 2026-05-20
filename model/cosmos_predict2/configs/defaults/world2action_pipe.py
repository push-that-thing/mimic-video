from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.config_world2action import SchedulerConfig, World2ActionPipelineConfig
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.world2action_dit import World2ActionDIT as VarNoiseWorld2ActionDIT
from imaginaire.lazy_config import LazyCall as L

ACTION_DECODER_NETS = {
    "lerobot": L(VarNoiseWorld2ActionDIT)(
        max_horizon=91,
        in_channels=6,
        out_channels=6,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    "so101": L(VarNoiseWorld2ActionDIT)(
        max_horizon=91,
        in_channels=6,
        out_channels=6,
        model_channels=512,
        num_blocks=12,
        num_heads=8,
        mlp_ratio=4.0,
        # torch SDPA backend: flash-attn has no Blackwell (sm_120) kernels.
        atten_backend="torch",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=512,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    # Short-horizon SO-101 variant. max_horizon=31 pairs with the slow
    # dataloader's action_lowdim_horizon=30 (1s of action prediction at 30 Hz).
    "slow_so101": L(VarNoiseWorld2ActionDIT)(
        max_horizon=31,
        in_channels=6,
        out_channels=6,
        model_channels=512,
        num_blocks=12,
        num_heads=8,
        mlp_ratio=4.0,
        # torch SDPA backend: flash-attn has no Blackwell (sm_120) kernels.
        atten_backend="torch",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=512,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    "libero": L(VarNoiseWorld2ActionDIT)(
        max_horizon=61,
        in_channels=10,
        out_channels=10,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    "bridge": L(VarNoiseWorld2ActionDIT)(
        max_horizon=16,
        in_channels=10,
        out_channels=10,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
}


def register_pipe() -> None:
    cs = ConfigStore.instance()

    for name, net in ACTION_DECODER_NETS.items():
        cs.store(
            group="world2action_pipe",
            package="world2action_pipe",
            name=name,
            node=L(World2ActionPipelineConfig)(
                precision="bfloat16",
                scheduler=SchedulerConfig(alpha=1.0, beta=1.0, num_denoising_steps=10),
                net=net,
                ema=EMAConfig(enabled=False),
            ),
        )
