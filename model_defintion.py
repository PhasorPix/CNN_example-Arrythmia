"""PyTorch network definition for binary heartbeat classification."""

from torch import nn


class ConvolutionalBlock1D(nn.Module):
    """Convolution, batch normalization, and GELU activation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.activation = nn.GELU()

    def forward(self, signal_bank):
        return self.activation(self.batch_norm(self.conv(signal_bank)))


class HeartbeatConvolutionalNetwork(nn.Module):
    """One-dimensional CNN producing one binary-classification logit per beat."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvolutionalBlock1D(1, 64, kernel_size=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            ConvolutionalBlock1D(64, 128, kernel_size=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            ConvolutionalBlock1D(128, 128, kernel_size=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
            ConvolutionalBlock1D(128, 128, kernel_size=5),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        # Retain eight coarse time regions instead of discarding temporal location.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, heartbeat_batch):
        features = self.features(heartbeat_batch)
        pooled_features = self.adaptive_pool(features)
        return self.classifier(pooled_features).squeeze(1)
