from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import torch
import numpy as np
import cv2

import os
from tqdm import tqdm
from enum import Enum

"""
┌────────────────┬────────┬──────────────┐
│ Visibility (m) │  Beta  │  Condition   │
├────────────────┼────────┼──────────────┤
│ 1000+          │ <0.004 │ Clear/haze   │
├────────────────┼────────┼──────────────┤
│ 500            │ 0.008  │ Light fog    │
├────────────────┼────────┼──────────────┤
│ 200            │ 0.02   │ Moderate fog │
├────────────────┼────────┼──────────────┤
│ 100            │ 0.04   │ Dense fog    │
├────────────────┼────────┼──────────────┤
│ 50             │ 0.08   │ Very dense   │
└────────────────┴────────┴──────────────┘
"""

class FogLevels(Enum):
    LIGHT = 500
    MODERATE = 200
    DENSE = 100
    HEAVY = 50

    @classmethod
    def from_level(cls, fog_level: int):
        try:
            return cls(fog_level).name.lower()
        except ValueError:
            return None

    @classmethod
    def from_label(cls, label: str):
        return cls[label.upper()].value    

class FogSynthesis:
    def __init__(self, model_name:str="LiheYoung/depth-anything-small-hf", 
                device:str="cuda", min_haze_level:FogLevels=FogLevels.LIGHT, 
                max_haze_level:FogLevels=FogLevels.HEAVY):

        self.image_preprocessor = AutoImageProcessor.from_pretrained(model_name)
        self.mono_depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)

        self.device = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device("cpu")        
        self.mono_depth_model.to(self.device)

        self.min_haze_level = min_haze_level
        self.max_haze_level = max_haze_level

    def generate_depth_map(self, image_arr:np.array):

        h, w = image_arr.shape[:2]

        inputs = self.image_preprocessor(images=image_arr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.mono_depth_model(**inputs)

        post_processed_output = self.image_preprocessor.post_process_depth_estimation(
            outputs,
            target_sizes=[(h, w)]
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]
        # depth = predicted_depth * 255 / predicted_depth.max()

        predicted_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min() + 1e-8)
        predicted_depth = 1.0 - predicted_depth # invert: sky/far → 1.0, road/near → 0.0
        return predicted_depth.detach().to(self.device)

    def sample_beta(self):
        visibility = np.random.uniform(self.min_haze_level.value, self.max_haze_level.value)
        beta = 3.912 / visibility
        return beta

    def spatial_beta_map(self, beta, H, W):

        y = torch.linspace(0, 1, H).view(H,1, 1)
        gradient = 1 + 0.5 * (1 - y)

        return beta * gradient

    def estimate_atmospheric_light(self, image, depth):

        # Sky = far depth regions
        sky_mask = depth > depth.quantile(0.9)

        if sky_mask.sum() < 10:
            return image.reshape(-1,3).mean(dim=0)

        sky_pixels = image[sky_mask.repeat(1,1,3)].reshape(-1,3)
        return sky_pixels.mean(dim=0)
    
    def synthesize_fog(self, image:torch.Tensor, depth:torch.Tensor, A:float=0.8):
        
        """
        clear_img: [B, 3, H, W] in [0, 1]
        depth_map: [B, 1, H, W] — estimated from monocular depth network
        beta: scattering coefficient (controls fog density)
        A: atmospheric light intensity

        Returns: foggy_img [B, 3, H, W]
        """

        if depth.ndim == 2:
            depth = depth.unsqueeze(-1) #(H, W, 1)

        # Normalize depth to realistic range [0, max_depth_meters]
        # Depth Anything outputs relative depth — scale to simulated meters
        max_depth = 150.0  # meters — typical max range for driving scenes
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = depth * max_depth  # now in [0, 150] meters            

        H, W, _ = image.shape

        #compute transmission map
        beta = self.sample_beta()
        beta_map = self.spatial_beta_map(beta, H, W).to(image.device)

        t = torch.exp(-beta_map * depth)        

        A = self.estimate_atmospheric_light(image, depth)

        foggy = image * t + (1 - t) * A
        return torch.clamp(foggy, 0.0, 1.0), t

    def save_image_tensor(self, tensor, save_path):

        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)

        img = tensor.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)

    def run(self, images_dir:str, output_dir:str):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.mono_depth_model.eval()        
        for image_fn in tqdm(os.listdir(images_dir)):
            image_path = os.path.join(images_dir, image_fn)

            image_arr = cv2.imread(image_path)
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)            

            depth_tensor = self.generate_depth_map(image_arr)

            image_tensor = torch.from_numpy(image_arr) / 255.0
            image_arr = image_tensor.to(self.device)

            foggy_tensor, _ = self.synthesize_fog(
                image=image_tensor,
                depth=depth_tensor
            )

            image_id = image_fn.split('.')[0]
            foggy_save_path = os.path.join(f'{output_dir}', f'{image_id}_foggy.png')

            self.save_image_tensor(
                foggy_tensor, foggy_save_path
            )

if __name__ == "__main__":
    
    source_dir = "/Users/shivvignesh/Documents/PersonalProjects/PanopticPerceptionProject/panoptic_perception/BDD100k/10k/train"
    output_dir = "/Users/shivvignesh/Documents/PersonalProjects/PanopticPerceptionProject/panoptic_perception/BDD100k/10k/foggy_train"
    
    fog_synthesis = FogSynthesis(min_haze_level=FogLevels.LIGHT, max_haze_level=FogLevels.DENSE)

    fog_synthesis.run(
        images_dir=source_dir,
        output_dir=output_dir
    )
                