import torch
import torch.nn as nn
import torchvision

#TODO, replace CustomVisionEncoder Class Name.
class CustomVisionEncoder(nn.Module):
    pass

class TorchVisionEncoder(torch.nn.Module):
    def __init__(self, model_name:str, latent_dim:int, pretrained:bool = False, tap_layers: list = None):
        super(TorchVisionEncoder, self).__init__()

        weights = "DEFAULT" if pretrained else None
        backbone = torchvision.models.get_model(model_name, weights=weights)

        # Remove the classification head, keep feature extractor
        if hasattr(backbone, "classifier"):
            # VGG, EfficientNet, MobileNet
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, latent_dim)
            self.feature_layers = backbone.features
        elif hasattr(backbone, "fc"):
            # ResNet, ResNeXt, etc.
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, latent_dim)
            self.feature_layers = nn.Sequential(*list(backbone.children())[:-1])
        elif hasattr(backbone, "head"):
            # Swin, ConvNeXt
            in_features = backbone.head.in_features
            backbone.head = nn.Linear(in_features, latent_dim)
            self.feature_layers = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Cannot identify head layer for {model_name}")

        self.backbone = backbone
        self.tap_layers = tap_layers
        self._tapped_features = {}
        self.hooks = []

        if tap_layers:
            for idx in tap_layers:
                hook = self.feature_layers[idx].register_forward_hook(
                    self._make_hook(idx)
                )
                self.hooks.append(hook)

            # dummy forward to determine tap layers channel
            tap_channels = self._infer_tap_channels()
            self.tap_projections = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(c, latent_dim)
                ) for c in tap_channels
            ])

    def _infer_tap_channels(self):
        """Run a dummy forward to discover channel dims at each tap layer."""
        self._tapped_features = {}
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.backbone(dummy)
        channels = [self._tapped_features[idx].shape[1] for idx in self.tap_layers]
        self._tapped_features = {}
        return channels

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            self._tapped_features[idx] = output
        return hook_fn

    def forward(self, x:torch.Tensor):
        self._tapped_features = {}
        latent = self.backbone(x)

        if self.tap_layers:
            features = [
                self.tap_projections[i](self._tapped_features[idx])
                for i, idx in enumerate(self.tap_layers)
            ]
            return latent, features

        return latent #(b, latent_dim)
    
#TODO    
CUSTOM_ENCODERS = {
    "CustomEncoder":CustomVisionEncoder()
}    

def build_vision_encoder(encoder_cfg: dict) -> nn.Module:
    """
    encoder_cfg = {
        "type": "vgg16",
        "latent_dim": 256, 
        "pretrained": True,
        "tap_layers": [4, 9, 16, 23, 30]  # optional, for MGDIP
    }
    """

    encoder_type = encoder_cfg["type"]
    latent_dim = encoder_cfg["latent_dim"]
    pretrained = encoder_cfg.get("pretrained", False)
    tap_layers = encoder_cfg.get("tap_layers", None)

    if encoder_type in CUSTOM_ENCODERS:
        # return CUSTOM_ENCODERS[encoder_type](latent_dim=latent_dim)
        raise NotImplementedError("Custom Vision Encoders are not supported yet!")

    if encoder_type in torchvision.models.list_models():
        return TorchVisionEncoder(encoder_type, latent_dim, pretrained, tap_layers)

    raise ValueError(f"Unknown encoder '{encoder_type}'")