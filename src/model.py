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
        # Reference CNN → (B⋅V, D_model, F', T')
        self.ref_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_ch, (3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(cnn_ch),
            nn.GELU(),
            nn.Conv2d(cnn_ch, d_model, (6, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        Fp_out = 1

        # After flattening the CNN frequency dimension, project to D_model
        # The input size to proj is C * Fp_out
        self.proj = nn.Linear(d_model * Fp_out, d_model)  # Corrected input size

        # Query tokens
        self.query_tokens = nn.Parameter(torch.randn(query_length, d_model))

        # Control MLP: (width, height) → 2 × D_model FiLM parameters
        self.ctrl_mlp = nn.Sequential(
            nn.Linear(2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        # Reference Control MLP for FiLMing reference features
        self.ref_ctrl_mlp = nn.Sequential(
            nn.Linear(2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        # Pure Encoder
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

        # Output regressor
        self.regressor = nn.Linear(d_model, feature_dim)

    def forward(
        self,
        refs: torch.Tensor,
        ref_controls: torch.Tensor,
        control: torch.Tensor,
    ):
        """
        Args:
          refs: (B, V, T, F)
          ref_controls: (B, V, 2) # [width, height] for each reference (in pixels)
          control: (B, 2)  # [width, height] (in pixels)
        Returns:
          sig: (B, T, F)
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

        # —— Apply FiLM to reference features x ——
        # ref_controls: (B, V, 2) -> (B*V, 2)
        ref_ctrl_input = ref_controls.view(B * V, 2)
        ref_ctrl_params = self.ref_ctrl_mlp(ref_ctrl_input)  # (B*V, 2*D_model)
        gamma_ref_ctrl, beta_ref_ctrl = ref_ctrl_params.chunk(
            2, dim=-1
        )  # Each (B*V, D_model)

        # Reshape x back to (B*V, Tp, D_model) for FiLM application
        x = x.view(B * V, Tp, C * Fp)
        x = gamma_ref_ctrl.unsqueeze(1) * x + beta_ref_ctrl.unsqueeze(1)  # Apply FiLM
        # Reshape x back to (B, V*Tp, D_model)
        x = x.view(B, V * Tp, C * Fp)

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
        sig = self.regressor(q_out)  # (B, T, F)

        return sig


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # feature_dim needs to be passed to SignSeq constructor
    model = SignSeq(feature_dim=6).to(device)
    model.eval()
    with (
        torch.no_grad(),
        profile(
            activities=(
                [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if torch.cuda.is_available()
                else [ProfilerActivity.CPU]
            ),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as prof,
    ):
        refs = torch.randn(2, 5, 4096, 6).to(device)
        ref_controls_tensor = torch.randint(50, 300, (2, 5, 2), dtype=torch.float32).to(
            device
        )
        control = torch.tensor([[100, 200], [150, 250]], dtype=torch.float32).to(device)
        output = model(refs, ref_controls_tensor, control)

        params = sum(p.numel() for p in model.parameters())

        print(model)
        print(f"Total parameters: {params:,} ")
        print(
            f"Input shape: {refs.shape}, Ref Controls shape: {ref_controls_tensor.shape}, Control shape: {control.shape}"
        )
        print(f"Output shape: {output.shape}")

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        )
    )
