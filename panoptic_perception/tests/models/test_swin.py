import pytest
import torch
import torch.nn as nn

from panoptic_perception.models.common import (
    PatchEmbed, ShiftedWindowMSA, SwinBlock, PatchMerge, SwinLayer
)

from panoptic_perception.models.swin_model import SwinBackbone, SwinClassifier, SwinObjectDetection

@pytest.fixture
def image_size():
    return (224, 224)

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def dummy_image(batch_size, image_size):
    h, w = image_size
    return torch.rand(batch_size, 3, h, w, dtype=torch.float32)

@pytest.fixture
def swin_cfg():
    return "panoptic_perception/configs/models/swin_model/swin_model_cls.cfg"

@pytest.fixture
def swin_yolov5_cfg():
    return "panoptic_perception/configs/models/swin_model/swin_model_yolov5.cfg"

def test_patch_embed(dummy_image, image_size):

    nc = dummy_image.shape[1]
    B = dummy_image.shape[0]
    patch_embed = PatchEmbed(image_size, nc)
        
    embed_dim = patch_embed.embed_dim
    kernel_size = patch_embed.patch_size
    stride = patch_embed.stride
    padding = patch_embed.padding

    def conv_out(h, k, s, p):
        return (h + 2*p - k) // s + 1
    
    h = conv_out(dummy_image.shape[-2], k=kernel_size, s=stride, p=padding)
    expected_shape = (B, h * h, embed_dim)

    patch = patch_embed(dummy_image)

    print(patch.shape, expected_shape)

def test_window_msa():

    def build_swin_attention_mask(input_res, window_size, shift_size, device=None):
        H, W = input_res
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        Wh, Ww = window_size

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -Wh), slice(-Wh, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -Ww), slice(-Ww, -shift_size), slice(-shift_size, None))

        region_id = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = region_id
                region_id += 1

        # window-partition: (1, H, W, 1) -> (nW, Wh, Ww, 1)
        nH, nW = H // Wh, W // Ww
        mask_windows = img_mask.view(1, nH, Wh, nW, Ww, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(-1, Wh * Ww)                      # (nW, Wh*Ww)

        # pairs with different region IDs must not attend
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask    

    embed_dim = 96
    window_size = (7, 7)
    input_res = (56, 56)
    num_pixels = input_res[0] * input_res[1] # 56x56
    num_heads = 3
    shift_size = 3
    batch_size = 2
    
    x = torch.rand(batch_size, num_pixels, embed_dim)
    x = x.reshape(-1, window_size[0] * window_size[1], embed_dim)

    msa = ShiftedWindowMSA(embed_dim, window_size, num_heads=num_heads)
    out = msa(x)
    print(out.shape, x.shape)

    mask = build_swin_attention_mask(input_res, window_size, shift_size)
    out = msa(x, mask)
    print(out.shape, x.shape)

def test_swin_block():

    embed_dim = 96
    window_size = 7
    input_res = (56, 56)
    num_pixels = input_res[0] * input_res[1] # 56x56
    num_heads = 3
    shift_size = 0
    batch_size = 2
    
    x = torch.rand(batch_size, num_pixels, embed_dim)

    swin_block = SwinBlock(
        embed_dim=embed_dim,
        input_res=input_res,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size
    )

    swin_block(x)

def test_patch_merge():

    embed_dim = 96
    window_size = 7
    input_res = (56, 56)
    num_pixels = input_res[0] * input_res[1] # 56x56
    num_heads = 3
    shift_size = 0
    batch_size = 2
    
    x = torch.rand(batch_size, num_pixels, embed_dim)

    patch_merge = PatchMerge(
        input_res=input_res,
        embed_dim=embed_dim
    )

    out = patch_merge(x)

    print(out.shape, x.shape)

def test_swin_layer():

    embed_dim = 96
    window_size = 7
    input_res = (56, 56)
    num_pixels = input_res[0] * input_res[1] # 56x56
    num_heads = 3
    shift_size = 0
    batch_size = 2
    depth = 2

    x = torch.rand(batch_size, num_pixels, embed_dim)

    swin_layer = SwinLayer(
        embed_dim=embed_dim,
        input_res=input_res,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        downsample=PatchMerge
    )

    out = swin_layer(x)
    print(out.shape, x.shape)

def test_swin_stack():
    embed_dim   = 96
    depths      = [2, 2, 6, 2]
    num_heads   = [3, 6, 12, 24]
    window_size = 7
    image_size  = (224, 224)
    batch_size  = 2
    drop_path_rate = 0.1

    # Stem: image -> patch tokens
    patch_embed = PatchEmbed(image_size, in_channels=3)
    x = patch_embed(torch.rand(batch_size, 3, *image_size))
    B, N, C = x.shape
    H = W = int(N ** 0.5)                      # 56 for 224/4
    assert (B, N, C) == (batch_size, H * W, embed_dim)

    # Per-block stochastic depth values (linspace 0 -> drop_path_rate),
    # sliced per stage by cumulative depth.
    dpr = [d.item() for d in torch.linspace(0, drop_path_rate, sum(depths))]

    num_layers = len(depths)
    patches_res = (H, W)

    layers = nn.ModuleList()
    for i_layer in range(num_layers):
        layer = SwinLayer(
            embed_dim  = int(embed_dim * 2 ** i_layer),
            input_res  = (patches_res[0] // (2 ** i_layer),
                          patches_res[1] // (2 ** i_layer)),
            depth      = depths[i_layer],
            num_heads  = num_heads[i_layer],
            window_size= window_size,
            drop_path  = dpr[sum(depths[:i_layer]) : sum(depths[:i_layer + 1])],
            downsample = PatchMerge if i_layer < num_layers - 1 else None,
        )
        layers.append(layer)

    # Forward + assert shapes at each stage boundary.
    expected = [
        (batch_size, 28 * 28, 192),   # after stage 0 (blocks at 56x56x96, then merge)
        (batch_size, 14 * 14, 384),   # after stage 1
        (batch_size,  7 *  7, 768),   # after stage 2
        (batch_size,  7 *  7, 768),   # after stage 3 (no downsample)
    ]

    for i, (layer, exp) in enumerate(zip(layers, expected)):
        x = layer(x)
        assert x.shape == exp, f"stage {i}: got {tuple(x.shape)}, expected {exp}"
        print(f"stage {i}: {tuple(x.shape)}")

def test_swin_stack_non_square():
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7
    image_size = (672, 1120) # H, W — was (224, 224)
    batch_size = 2

    patch_embed = PatchEmbed(image_size, in_channels=3)
    x = patch_embed(torch.rand(batch_size, 3, *image_size))
    B, N, C = x.shape

    # patches_res is a TUPLE now, not a scalar
    H, W = image_size[0] // patch_embed.patch_size, image_size[1] // patch_embed.patch_size
    assert (B, N, C) == (batch_size, H * W, embed_dim)
    patches_res = (H, W)

    num_layers = len(depths)
    layers = nn.ModuleList()
    for i_layer in range(num_layers):
        layer = SwinLayer(
            embed_dim  = int(embed_dim * 2 ** i_layer),
            input_res  = (patches_res[0] // (2 ** i_layer),
                          patches_res[1] // (2 ** i_layer)),
            depth      = depths[i_layer],
            num_heads  = num_heads[i_layer],
            window_size= window_size,
            downsample = PatchMerge if i_layer < num_layers - 1 else None,
        )
        layers.append(layer)

    # Expected shapes derived from (H, W) directly.
    def stage_shape(i, has_downsample):
        h = patches_res[0] // (2 ** (i + (1 if has_downsample else 0)))
        w = patches_res[1] // (2 ** (i + (1 if has_downsample else 0)))
        c = embed_dim * 2 ** (i + (1 if has_downsample else 0))
        return (batch_size, h * w, c)

    expected = [stage_shape(i, has_downsample=(i < num_layers - 1)) for i in range(num_layers)]

    for i, (layer, exp) in enumerate(zip(layers, expected)):
        x = layer(x)
        assert x.shape == exp, f"stage {i}: got {tuple(x.shape)}, expected {exp}"
        print(f"stage {i}: {tuple(x.shape)}")


def test_swin_stack_nn(swin_cfg):

    backbone = SwinBackbone(swin_cfg)
    x = torch.rand(2, 3, 224, 224)
    final, taps = backbone(x, intercept_layers={0, 1, 2, 3})
    assert final.shape == (2, 49, backbone.final_channels)
    assert set(taps.keys()) == {0, 1, 2, 3}

    for idx, tap in taps.items():
        print(idx, tap.shape)

def test_swin_classifier(swin_cfg):

    model = SwinClassifier.from_config(swin_cfg)
    x = torch.rand(2, 3, 224, 224)
    model_outputs = model(x)
    assert model_outputs.logits.shape == (2, model.head.out_features)

def test_swin_yolov5(swin_yolov5_cfg):

    model = SwinObjectDetection.from_config(swin_yolov5_cfg)
    x = torch.rand(2, 3, 672, 1120)

    model(x)