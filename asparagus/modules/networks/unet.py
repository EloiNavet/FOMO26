import torch
import torch.nn as nn
from asparagus.functional.representations import build_h_global
from gardening_tools.modules.networks.components.blocks import MultiLayerConvDropoutNormNonlin
from gardening_tools.modules.networks.unet import UNet, UNetCLSREG


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
        return x * (1.0 + gamma.view(shape)) + beta.view(shape)


class UNetSSL(UNet):
    supports_reconstruction = True
    supports_multiscale_features = True
    supports_segmentation = True
    supports_tokens = False  # CNN backbone: no native patch/token seam, so masking is applied on the input
    pretrained_backbone_prefixes = ("encoder.",)

    def __init__(
        self,
        *args,
        head_out_dim: int = 256,
        head_hidden_dim: int = 512,
        num_modalities: int = 15,
        modality_embedding_dim: int = 64,
        modality_conditioning: bool = True,
        h_global_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_feature_channels = tuple(self._encoder_channels())
        self.global_feature_dim = sum(self.encoder_feature_channels)
        self.h_global_layer_norm_enabled = bool(h_global_layer_norm)
        self.h_global_norm = nn.LayerNorm(self.global_feature_dim) if self.h_global_layer_norm_enabled else nn.Identity()
        self.head_demo = ProjectionHead(self.global_feature_dim, head_hidden_dim, head_out_dim)
        self.head_patho = ProjectionHead(self.global_feature_dim, head_hidden_dim, head_out_dim)
        self.head_stage1_anatomy = ProjectionHead(self.global_feature_dim, head_hidden_dim, head_out_dim)
        self.num_modalities = int(num_modalities)
        self.unknown_modality_id = self.num_modalities
        self.modality_conditioning = bool(modality_conditioning)
        self.modality_embedding = nn.Embedding(self.num_modalities + 1, modality_embedding_dim)
        self.encoder_films = nn.ModuleList(
            [ModalityFiLM(channels, modality_embedding_dim) for channels in self._encoder_channels()]
        )

    def _encoder_channels(self) -> list[int]:
        filters = self.encoder.filters
        return [filters, filters * 2, filters * 4, filters * 8, filters * 16]

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
        modality_embedding = None
        if self.modality_conditioning and use_modality_conditioning:
            modality_embedding = self._modality_embedding(modality_id, x.shape[0], x.device)

        x0 = self.encoder.in_conv(x)
        if modality_embedding is not None:
            x0 = self.encoder_films[0](x0, modality_embedding)

        x1 = self.encoder.pool1(x0)
        x1 = self.encoder.encoder_conv1(x1)
        if modality_embedding is not None:
            x1 = self.encoder_films[1](x1, modality_embedding)

        x2 = self.encoder.pool2(x1)
        x2 = self.encoder.encoder_conv2(x2)
        if modality_embedding is not None:
            x2 = self.encoder_films[2](x2, modality_embedding)

        x3 = self.encoder.pool3(x2)
        x3 = self.encoder.encoder_conv3(x3)
        if modality_embedding is not None:
            x3 = self.encoder_films[3](x3, modality_embedding)

        x4 = self.encoder.pool4(x3)
        x4 = self.encoder.encoder_conv4(x4)
        if modality_embedding is not None:
            x4 = self.encoder_films[4](x4, modality_embedding)

        return [x0, x1, x2, x3, x4]

    def forward_with_features(self, x: torch.Tensor, modality_id=None):
        representations = self.encode_representations(x, modality_id=modality_id)
        output = self.decoder(representations["h_dense"])
        return output, representations["h_global"]

    def forward_encoder_only(self, x: torch.Tensor, modality_id=None) -> torch.Tensor:
        return self._encode_skips(x, modality_id=modality_id)[-1]

    def forward_encoder_to_level(self, x: torch.Tensor, level: int = -1, modality_id=None) -> torch.Tensor:
        """Run the U-Net encoder only through the requested shallow-to-deep feature level."""
        n_stages = len(self.encoder_feature_channels)
        stage_index = int(level) if int(level) >= 0 else n_stages + int(level)
        if not 0 <= stage_index < n_stages:
            raise ValueError(f"feature_level={level} out of range for {n_stages} U-Net encoder stages.")

        embedding = None
        if self.modality_conditioning:
            embedding = self._modality_embedding(modality_id, x.shape[0], x.device)
        features = self.encoder.in_conv(x)
        if embedding is not None:
            features = self.encoder_films[0](features, embedding)
        for index in range(1, stage_index + 1):
            features = getattr(self.encoder, f"pool{index}")(features)
            features = getattr(self.encoder, f"encoder_conv{index}")(features)
            if embedding is not None:
                features = self.encoder_films[index](features, embedding)
        return features

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


def unet_b_lw_dec(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    use_skip_connections: bool = True,
    **_unused,
):
    return UNet(
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
        use_skip_connections=use_skip_connections,
    )


def unet_b(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
    **_unused,
):
    return UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
        deep_supervision=deep_supervision,
    )


def unet_m(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
    use_skip_connections: bool = True,
    head_out_dim: int = 256,
    head_hidden_dim: int = 512,
    num_modalities: int = 15,
    modality_embedding_dim: int = 64,
    modality_conditioning: bool = True,
    h_global_layer_norm: bool = False,
    **_unused,
):
    return UNetSSL(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=64,
        deep_supervision=deep_supervision,
        use_skip_connections=use_skip_connections,
        head_out_dim=head_out_dim,
        head_hidden_dim=head_hidden_dim,
        num_modalities=num_modalities,
        modality_embedding_dim=modality_embedding_dim,
        modality_conditioning=modality_conditioning,
        h_global_layer_norm=h_global_layer_norm,
    )


def unet_clsreg_b(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
    **_unused,
):
    return UNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=32,
        deep_supervision=deep_supervision,
    )


def unet_clsreg_m(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
    **_unused,
):
    # Width-64 cls/reg head matching the unet_m (UNetSSL, starting_filters=64) encoder,
    # so encoder-only transfer from our UNet-M SSL checkpoints lines up shape-for-shape.
    return UNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=64,
        deep_supervision=deep_supervision,
    )


def unet_clsreg_s(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    **_unused,
):
    return UNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        starting_filters=16,
    )


def unet_tiny(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    deep_supervision: bool = False,
    use_skip_connections: bool = True,
    **_unused,
):
    return UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=2,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        deep_supervision=deep_supervision,
        use_skip_connections=use_skip_connections,
    )


def unet_clsreg_tiny(
    input_channels: int = 1,
    output_channels: int = 1,
    dimensions: str = "3D",
    **_unused,
):
    return UNetCLSREG(
        input_channels=input_channels,
        output_channels=output_channels,
        dimensions=dimensions,
        starting_filters=2,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
    )


if __name__ == "__main__":
    model = unet_b_lw_dec(input_channels=1, output_channels=1)
    print(model)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    print(y.shape)
