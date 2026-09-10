import logging
import torch
from asparagus.functional.representations import build_h_global
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.blocks import ResidualBlock
from gardening_tools.modules.networks.components.encoders import ResidualUNetEncoder
from gardening_tools.modules.networks.components.heads import ClsRegHead
from gardening_tools.modules.networks.resunet import ResidualEncoderUNet
from torch import nn
from typing import List, Tuple, Type, Union


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityFiLM(nn.Module):
    def __init__(self, channels: int, embedding_dim: int):
        super().__init__()
        self.affine = nn.Linear(embedding_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, modality_embedding: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.affine(modality_embedding)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        gamma = gamma.view(shape)
        beta = beta.view(shape)
        return x * (1.0 + gamma) + beta


class ResidualEncoderUNetCLSREG(BaseNet):
    """Late-fusion classification/regression network.

    Each input modality (channel group) is processed independently through a shared encoder.
    The resulting features are concatenated along the channel dimension and passed through
    a ClsRegHead (global pool + linear).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str,
        kernel_size: int,
        stride: int,
        features_per_stage: list,
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = True,
        encoder_basic_block: Type[ResidualBlock] = ResidualBlock,
        decoder: nn.Module = ClsRegHead,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        late_fusion: bool = False,
        encoder_pool: str = "max",
    ):
        super().__init__()

        # Extract dropout rates from kwargs
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        encoder_dropout_rate = dropout_op_kwargs.get("encoder_dropout_rate", 0.0)
        decoder_dropout_rate = dropout_op_kwargs.get("decoder_dropout_rate", 0.0)
        inplace = dropout_op_kwargs.get("inplace", True)

        encoder_pool = str(encoder_pool).strip().lower()
        if encoder_pool not in {"avg", "max"}:
            raise ValueError(f"encoder_pool must be 'avg' or 'max', got {encoder_pool!r}.")
        if dimensions == "2D":
            conv_op = nn.Conv2d
            norm_op = nn.InstanceNorm2d
            pool_op = nn.AvgPool2d if encoder_pool == "avg" else nn.MaxPool2d
            clsreg_pool_op = nn.AdaptiveAvgPool2d
            if encoder_dropout_rate > 0.0:
                dropout_op = nn.Dropout2d
        elif dimensions == "3D":
            conv_op = nn.Conv3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.AvgPool3d if encoder_pool == "avg" else nn.MaxPool3d
            clsreg_pool_op = nn.AdaptiveAvgPool3d
            if encoder_dropout_rate > 0.0:
                dropout_op = nn.Dropout3d
        else:
            logging.warning("Uuh, dimensions not in ['2D', '3D']")

        self.num_classes = output_channels
        self.late_fusion = late_fusion
        self.encoder_pool = encoder_pool
        self.pretrained_backbone_prefixes = ("encoder.",)

        self.encoder = ResidualUNetEncoder(
            input_channels=1 if late_fusion else input_channels,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_size=kernel_size,
            stride=stride,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs={"p": encoder_dropout_rate, "inplace": inplace},
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=encoder_basic_block,
            pool_op=pool_op,
        )

        self.decoder = decoder(
            pool_op=clsreg_pool_op,
            input_channels=features_per_stage[-1] * input_channels if late_fusion else features_per_stage[-1],
            output_channels=output_channels,
            dropout_rate=decoder_dropout_rate,
        )

    def _encode(self, x):
        if not self.late_fusion:
            return self.encoder(x)  # early-fusion of modalities

        # late-fusion of modalities
        B, N = x.shape[:2]
        skips = self.encoder(x.view(B * N, -1, *x.shape[2:]))
        return [s.view(B, N * s.shape[1], *s.shape[2:]) for s in skips]

    def forward(self, x):
        skips = self._encode(x)
        return self.decoder(skips)

    def forward_with_features(self, x):
        skips = self._encode(x)
        output = self.decoder(skips)
        return output, skips[-1]

    def forward_encoder_only(self, x):
        skips = self._encode(x)
        return skips[-1]

    def freeze_backbone(self):
        """Freeze the encoder backbone for linear probing."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()


class ResidualEncoderUNetSSL(ResidualEncoderUNet):
    supports_reconstruction = True
    supports_multiscale_features = True
    supports_segmentation = True
    supports_tokens = False  # CNN backbone: no native patch/token seam, so masking is applied on the input
    pretrained_backbone_prefixes = ("encoder.",)

    def __init__(
        self,
        dimensions: str,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        features_per_stage: list,
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = True,
        encoder_basic_block: Type[ResidualBlock] = ResidualBlock,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision: bool = False,
        use_skip_connections: bool = True,
        head_out_dim: int = 256,
        head_hidden_dim: int = 512,
        num_modalities: int = 15,
        modality_embedding_dim: int = 64,
        modality_conditioning: bool = True,
        decoder_modality_conditioning: bool = False,
        h_global_layer_norm: bool = False,
    ):
        super().__init__(
            dimensions=dimensions,
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            features_per_stage=features_per_stage,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            encoder_basic_block=encoder_basic_block,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            use_skip_connections=use_skip_connections,
        )

        self.encoder_feature_channels = tuple(int(channels) for channels in features_per_stage)
        self.global_feature_dim = sum(self.encoder_feature_channels)
        self.h_global_layer_norm_enabled = bool(h_global_layer_norm)
        self.h_global_norm = nn.LayerNorm(self.global_feature_dim) if self.h_global_layer_norm_enabled else nn.Identity()

        self.head_demo = ProjectionHead(self.global_feature_dim, head_hidden_dim, head_out_dim)
        self.head_patho = ProjectionHead(self.global_feature_dim, head_hidden_dim, head_out_dim)
        self.head_stage1_anatomy = ProjectionHead(self.global_feature_dim, head_hidden_dim, head_out_dim)
        self.num_modalities = int(num_modalities)
        self.unknown_modality_id = self.num_modalities
        self.modality_conditioning = bool(modality_conditioning)
        self.decoder_modality_conditioning = bool(decoder_modality_conditioning)
        self.modality_embedding = nn.Embedding(self.num_modalities + 1, modality_embedding_dim)
        self.encoder_films = nn.ModuleList([ModalityFiLM(channels, modality_embedding_dim) for channels in features_per_stage])
        self.decoder_films = nn.ModuleList([ModalityFiLM(channels, modality_embedding_dim) for channels in features_per_stage])

    def _normalize_h_global(self, h_global: torch.Tensor) -> torch.Tensor:
        return self.h_global_norm(h_global)

    def _modality_ids(self, modality_id, batch_size: int, device: torch.device) -> torch.Tensor:
        if modality_id is None:
            ids = torch.full((batch_size,), self.unknown_modality_id, dtype=torch.long, device=device)
        elif isinstance(modality_id, torch.Tensor):
            ids = modality_id.to(device=device, dtype=torch.long).view(-1)
        else:
            ids = torch.as_tensor(modality_id, dtype=torch.long, device=device).view(-1)

        if ids.numel() == 1 and batch_size > 1:
            ids = ids.expand(batch_size)
        ids = ids.clone()
        invalid = (ids < 0) | (ids >= self.num_modalities)
        ids[invalid] = self.unknown_modality_id
        return ids

    def _modality_embedding(self, modality_id, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.modality_embedding(self._modality_ids(modality_id, batch_size, device))

    def _encode_skips(self, x: torch.Tensor, modality_id=None, use_modality_conditioning: bool = True):
        if not self.modality_conditioning or not use_modality_conditioning:
            return self.encoder(x)

        modality_embedding = self._modality_embedding(modality_id, x.shape[0], x.device)
        skips = []
        features = self.encoder.stem(x)
        for stage_idx, stage in enumerate(self.encoder.stages):
            features = stage(features)
            features = self.encoder_films[stage_idx](features, modality_embedding)
            skips.append(features)
        return skips

    def decode_from_skips(self, skips, modality_id=None):
        conditioned_skips = skips
        if self.modality_conditioning and self.decoder_modality_conditioning and modality_id is not None:
            device = skips[-1].device
            batch_size = skips[-1].shape[0]
            modality_embedding = self._modality_embedding(modality_id, batch_size, device)
            conditioned_skips = [film(skip, modality_embedding) for film, skip in zip(self.decoder_films, skips)]
        return self.decoder(conditioned_skips)

    def forward_with_features(self, x: torch.Tensor, modality_id=None):
        representations = self.encode_representations(x, modality_id=modality_id)
        output = self.decode_from_skips(representations["h_dense"], modality_id=modality_id)
        return output, representations["h_global"]

    def encode_pooled(self, x: torch.Tensor, modality_id=None) -> torch.Tensor:
        return self.encode_representations(x, modality_id=modality_id)["h_global"]

    def encode_representations(
        self,
        x: torch.Tensor,
        modality_id=None,
        use_modality_conditioning: bool = True,
    ) -> dict:
        h_dense = self._encode_skips(
            x,
            modality_id=modality_id,
            use_modality_conditioning=use_modality_conditioning,
        )
        h_global = self._normalize_h_global(build_h_global(h_dense))
        output = {"h_dense": h_dense, "h_global": h_global}
        return output

    def forward_encoder_only(self, x: torch.Tensor, modality_id=None) -> torch.Tensor:
        return self._encode_skips(x, modality_id=modality_id)[-1]

    def forward_encoder_to_level(self, x: torch.Tensor, level: int = -1, modality_id=None) -> torch.Tensor:
        """Encode only up to a chosen encoder stage and return that feature-map grid.

        ``level`` indexes the encoder stages (negative allowed; ``-1`` = bottleneck). Stages deeper
        than ``level`` are **not executed**, so a caller that also freezes them keeps the latent at a
        higher spatial resolution without leaving deeper stages as DDP "unused parameters".
        """
        n_stages = len(self.encoder.stages)
        k = level if level >= 0 else n_stages + level
        if not 0 <= k < n_stages:
            raise ValueError(f"feature_level={level} out of range for {n_stages} encoder stages.")
        if k == n_stages - 1:
            return self.forward_encoder_only(x, modality_id=modality_id)

        use_conditioning = bool(self.modality_conditioning)
        modality_embedding = self._modality_embedding(modality_id, x.shape[0], x.device) if use_conditioning else None
        features = self.encoder.stem(x)
        for stage_idx in range(k + 1):
            features = self.encoder.stages[stage_idx](features)
            if use_conditioning:
                features = self.encoder_films[stage_idx](features, modality_embedding)
        return features


# Encoder 29M parameters
# Full model 42M parameters
# This is the "classic" unet, but with residual encoder blocks
def resenc_unet_s(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    **_unused,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(2, 2, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )


# Encoder 90M parameters
# Full model 102M parameters
def resenc_unet_b(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    use_skip_connections=True,
    **_unused,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
        use_skip_connections=use_skip_connections,
    )


def resenc_unet_b_ssl(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    use_skip_connections=True,
    head_out_dim: int = 256,
    head_hidden_dim: int = 512,
    num_modalities: int = 15,
    modality_embedding_dim: int = 64,
    modality_conditioning: bool = True,
    decoder_modality_conditioning: bool = False,
    h_global_layer_norm: bool = False,
):
    return ResidualEncoderUNetSSL(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
        use_skip_connections=use_skip_connections,
        head_out_dim=head_out_dim,
        head_hidden_dim=head_hidden_dim,
        num_modalities=num_modalities,
        modality_embedding_dim=modality_embedding_dim,
        modality_conditioning=modality_conditioning,
        decoder_modality_conditioning=decoder_modality_conditioning,
        h_global_layer_norm=h_global_layer_norm,
    )


# Encoder 90M parameters
# Full model - 90.3 M    Total params
def resenc_unet_b_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    dropout_op_kwargs: dict = None,
    late_fusion: bool = False,
    encoder_pool: str = "max",
):
    # input_channels is the number of modalities (e.g. 2 for T1+T2).
    # Each modality is a single-channel volume, so the encoder always uses input_channels=1.
    return ResidualEncoderUNetCLSREG(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        dropout_op_kwargs=dropout_op_kwargs,
        late_fusion=late_fusion,
        encoder_pool=encoder_pool,
    )


# Encoder 345M parameters
# Full model 391M parameters
def resenc_unet_l(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    **_unused,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(64, 128, 256, 512, 620, 620),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )


def resenc_unet_l_clsreg(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision=False,
    dropout_op_kwargs: dict = None,
    encoder_pool: str = "max",
):
    # input_channels is the number of modalities (e.g. 2 for T1+T2).
    # Each modality is a single-channel volume, so the encoder always uses input_channels=1.
    return ResidualEncoderUNetCLSREG(
        dimensions=dimensions,
        input_channels=1,
        num_modalities=input_channels,
        output_channels=output_channels,
        features_per_stage=(64, 128, 256, 512, 620, 620),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        dropout_op_kwargs=dropout_op_kwargs,
        encoder_pool=encoder_pool,
    )


# Encoder 602M parameters
# Full model 662M parameters
def resenc_unet_h(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    **_unused,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(64, 128, 256, 512, 768, 768),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 8, 8),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )


# Encoder 989M parameters
# Full model 1079M parameters
def resenc_unet_g(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    **_unused,
):
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=(64, 128, 256, 512, 1024, 1024),
        stride=2,
        kernel_size=3,
        n_blocks_per_stage=(1, 3, 4, 6, 8, 8),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        deep_supervision=deep_supervision,
    )


# Debug-only geometry. Not a canonical architecture: nothing is trained, released or
# deserialized through it, and no checkpoint records these shapes.
#
# `ResidualUNetDecoder` is not a loop. It writes out `upsample1`..`upsample5` and
# `decoder_conv1`..`decoder_conv5` by hand, indexing `features_per_stage[0..5]` and
# `n_conv_per_stage[0..4]`. Six encoder stages and five decoder stages are therefore a
# structural requirement of the architecture, not a size choice, and every canonical factory in
# this module already passes exactly that. This debug factory used to pass four and three, so it
# raised `IndexError: tuple index out of range` at `upsample4` before any caller could run.
_DEBUG_FEATURES_PER_STAGE = (4, 8, 4, 4, 4, 4)
_DEBUG_N_BLOCKS_PER_STAGE = (1,) * 6
_DEBUG_N_CONV_PER_STAGE_DECODER = (1,) * 5


def resenc_unet_debug(
    dimensions,
    input_channels,
    output_channels,
    deep_supervision=False,
    stride=None,
):
    """Smallest ResidualEncoderUNet this decoder can build, ~15k parameters.

    The stride is derived from `deep_supervision` because the two are not independent:

    * Deep supervision compares five decoder outputs against a label pyramid whose factors are
      fixed at ``(1, 1/2, 1/4, 1/8, 1/16, 1/16)``. Only a stride-2 network produces logits at
      those resolutions, so ``deep_supervision=True`` needs ``stride=2`` -- and that downsamples
      by 32, so its callers must supply at least 64^3 or ``InstanceNorm3d`` refuses the
      single-voxel bottleneck in training mode.
    * Without deep supervision nothing constrains the resolutions, and the callers use 16^3 and
      32^3 volumes. ``stride=1`` keeps the network shape-preserving so those stay cheap.

    Deriving it rather than hard-coding one value is what lets both kinds of caller work from a
    single factory. `stride` may still be passed explicitly to override the default; the rest of
    the signature is unchanged.
    """
    if stride is None:
        stride = 2 if deep_supervision else 1
    return ResidualEncoderUNet(
        dimensions=dimensions,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=_DEBUG_FEATURES_PER_STAGE,
        stride=stride,
        kernel_size=3,
        n_blocks_per_stage=_DEBUG_N_BLOCKS_PER_STAGE,
        n_conv_per_stage_decoder=_DEBUG_N_CONV_PER_STAGE_DECODER,
        deep_supervision=deep_supervision,
    )
