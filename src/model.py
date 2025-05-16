import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torchaudio.models import Conformer


class SignSeq(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        feature_dim: int = 6,  # F = [x_rel, y_rel, force, altitude, azimuth_sin, azimuth_cos]
        query_length: int = 4096,  # T
        cnn_ch: int = 32,
        d_model: int = 256,
        n_heads: int = 4,
        ffn_dim: int = 512,
        n_layers: int = 6,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1) Reference CNN → (B⋅V, D_model, F', T')
        self.ref_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_ch, (3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(cnn_ch),
            nn.GELU(),
            nn.Conv2d(cnn_ch, d_model, (3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        # Calculate the output height dimension after the CNN
        # Input height is feature_dim (F)
        # Stride in height is 1 for both conv layers
        # Kernel size in height is 3 for both conv layers
        # Padding in height is 1 for both conv layers
        # Output height after 1st conv: floor((F + 2*1 - 3)/1) + 1 = floor(F - 1) + 1
        # Output height after 2nd conv: floor((floor(F - 1) + 1 + 2*1 - 3)/1) + 1 = floor(F - 1) + 1
        # For F=6, Fp = floor(6 - 1) + 1 = 5 + 1 = 6
        # The output feature dimension Fp should be feature_dim itself with stride 1 and kernel/padding 3,1
        Fp_out = feature_dim  # Based on calculation

        # After flattening the CNN frequency dimension, project to D_model
        # The input size to proj is C * Fp_out
        self.proj = nn.Linear(d_model * Fp_out, d_model)  # Corrected input size

        # 2) Query tokens
        self.query_tokens = nn.Parameter(torch.randn(query_length, d_model))

        # 3) Control MLP: (width, height) → 2 × D_model FiLM parameters
        self.ctrl_mlp = nn.Sequential(
            nn.Linear(2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        # 4) Pure Encoder
        self.encoder = Conformer(
            input_dim=d_model,
            num_heads=n_heads,
            ffn_dim=ffn_dim,
            num_layers=n_layers,
            depthwise_conv_kernel_size=kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )

        # 5) Output normalized x, y in [0,1]
        self.regressor = nn.Linear(d_model, feature_dim)

    def forward(
        self,
        refs: torch.Tensor,
        control: torch.Tensor,
    ):
        """
        Args:
          refs: (B, V, T, F)
          control: (B, 2)  # [width, height] (in pixels)
        Returns:
          sig: (B, T, F)  # sig[...,0] and sig[...,1] have been multiplied by the corresponding width/height
        """
        B, V, T, F = refs.shape
        # —— 1) Encode all reference signatures ——
        # Input to CNN is (B*V, 1, F, T)
        x = refs.view(B * V, 1, F, T)
        # Output from CNN is (B*V, D_model, F', T')
        x = self.ref_cnn(x)
        _, C, Fp, Tp = x.shape
        # Reshape to (B*V, Tp, C*Fp) for the linear layer
        x = x.permute(0, 3, 1, 2).reshape(B * V, Tp, C * Fp)
        # Linear projection to (B*V, Tp, D_model)
        x = self.proj(x)
        # Reshape to (B, V*Tp, D_model)
        x = x.view(B, V * Tp, -1)

        # —— 2) Prepare query tokens and control FiLM parameters ——
        # Query: (B, T, D_model)
        q = self.query_tokens.unsqueeze(1).expand(-1, B, -1).transpose(0, 1)

        # Control FiLM parameters
        ctrl_params = self.ctrl_mlp(control)  # (B, 2*D_model)
        gamma_ctrl, beta_ctrl = ctrl_params.chunk(2, dim=-1)  # Each (B, D_model)

        # Apply FiLM to query
        q = gamma_ctrl.unsqueeze(1) * q + beta_ctrl.unsqueeze(1)

        # —— 3) Concatenate references + queries, enter pure Encoder ——
        enc_in = torch.cat([x, q], dim=1)  # (B, V*Tp + T, D_model)

        # Calculate the lengths for the concatenated tensor
        # The references part has length Tp for each of the V references in each batch
        # The queries part has length T for each batch
        # The total length for each item in the batch is V * Tp + T
        enc_in_lengths = torch.full(
            (B,),
            V * Tp + T,
            dtype=torch.int64,
            device=enc_in.device,  # Ensure lengths tensor is on the same device as input
        )

        # Pass the input tensor and the lengths tensor to the encoder
        enc_out, _ = self.encoder(enc_in, enc_in_lengths)  # (B, V*Tp + T, D_model)

        # —— 4) Extract query output & regress ——
        q_out = enc_out[:, x.size(1) :, :]  # (B, T, D_model)
        sig = self.regressor(q_out)  # (B, T, F)  # x,y in [0,1] assumed

        return sig


if __name__ == "__main__":
    model = SignSeq()
    model.eval()
    with torch.no_grad():
        refs = torch.randn(2, 5, 4096, 6)  # (B, V, T, F)
        control = torch.tensor(
            [[100, 200], [150, 250]], dtype=torch.float32
        )  # (B, 2) [width, height]
        output = model(refs, control)

        params = sum(p.numel() for p in model.parameters())

        print(model)
        print(f"Total parameters: {params:,} ")
        print(f"Input shape: {refs.shape}, Control shape: {control.shape}")
        print(f"Output shape: {output.shape}")
