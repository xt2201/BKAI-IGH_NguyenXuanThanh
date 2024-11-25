import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Union, List
import timm

class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        gap = self.gap(x).view(b, c)
        attention = self.attention(gap).view(b, c, 1, 1)
        return x * attention.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.attention:
            x = self.attention(x)
        x = self.dropout(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        num_classes: int = 2,
        decoder_attention: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=(encoder_weights is not None),
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get encoder channels
        encoder_channels = self.encoder.feature_info.channels()

        # Decoder blocks
        self.decoder4 = ConvBlock(
            encoder_channels[4] + encoder_channels[3], 512, decoder_attention, dropout_rate
        )
        self.decoder3 = ConvBlock(
            512 + encoder_channels[2], 256, decoder_attention, dropout_rate
        )
        self.decoder2 = ConvBlock(
            256 + encoder_channels[1], 128, decoder_attention, dropout_rate
        )
        self.decoder1 = ConvBlock(
            128 + encoder_channels[0], 64, decoder_attention, dropout_rate
        )

        # Final classification
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        encoder_features = self.encoder(x)

        # Decoder
        x = self.decoder4(
            torch.cat(
                [
                    F.interpolate(
                        encoder_features[4],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    encoder_features[3],
                ],
                dim=1,
            )
        )

        x = self.decoder3(
            torch.cat(
                [
                    F.interpolate(
                        x,
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    encoder_features[2],
                ],
                dim=1,
            )
        )

        x = self.decoder2(
            torch.cat(
                [
                    F.interpolate(
                        x,
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    encoder_features[1],
                ],
                dim=1,
            )
        )

        x = self.decoder1(
            torch.cat(
                [
                    F.interpolate(
                        x,
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    encoder_features[0],
                ],
                dim=1,
            )
        )

        # Final upsampling and classification
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.final_conv(x)

        return x

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)

        if target.shape[1] != pred.shape[1]:
            target = target.permute(0, 3, 1, 2)  # Convert [B, H, W, C] to [B, C, H, W]

        # Compute Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        dice_numerator = 2 * (pred_sigmoid * target).sum(dim=(2, 3)) + self.smooth
        dice_denominator = (pred_sigmoid.pow(2) + target.pow(2)).sum(dim=(2, 3)) + self.smooth
        dice_loss = 1 - (dice_numerator / dice_denominator).mean()

        # Compute BCE Loss
        bce_loss = self.bce(pred, target)

        # Return combined loss
        return 0.5 * dice_loss + 0.5 * bce_loss

class DiceCoefficient:
    def __init__(self, smooth: float = 1e-6):
        self.smooth = smooth

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred) > 0.5
        pred = pred.float()

        numerator = 2 * (pred * target).sum(dim=(2, 3)) + self.smooth
        denominator = (pred + target).sum(dim=(2, 3)) + self.smooth

        return (numerator / denominator).mean()

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer_params: dict,
        scheduler_params: dict
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = DiceBCELoss()
        self.metric = DiceCoefficient()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            **optimizer_params
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            **scheduler_params
        )

    def train_step(self, dataloader):
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0

        for batch in dataloader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            dice_score = self.metric(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_score.item()

        return epoch_loss / len(dataloader), epoch_dice / len(dataloader)

    @torch.no_grad()
    def validate_step(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        epoch_dice = 0

        for batch in dataloader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            dice_score = self.metric(outputs, masks)

            epoch_loss += loss.item()
            epoch_dice += dice_score.item()

        return epoch_loss / len(dataloader), epoch_dice / len(dataloader)

def get_model_components(config):
    model = UNet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=2,
        decoder_attention=True,
        dropout_rate=config.DROPOUT_RATE,
    )

    # Optimizer parameters
    optimizer_params = {
        "lr": config.LEARNING_RATE,
        "weight_decay": 1e-5
    }

    # Scheduler parameters
    scheduler_params = {
        "mode": "min",
        "factor": config.SCHEDULER_FACTOR,
        "patience": config.SCHEDULER_PATIENCE,
        "min_lr": config.SCHEDULER_MIN_LR,
        "verbose": True
    }

    trainer = ModelTrainer(
        model=model,
        device=config.DEVICE,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params
    )

    return trainer