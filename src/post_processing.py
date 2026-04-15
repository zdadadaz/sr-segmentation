import numpy as np
import cv2
import torch
import noise
from scipy.ndimage import gaussian_filter, sobel

class VegetationNoiseSynthesizer:
    """
    Synthesize Perlin/Simplex noise textures in vegetation regions after SR.
    """

    def __init__(
        self,
        noise_type: str = "simplex",  # "perlin" or "simplex"
        octaves: int = 6,
        lacunarity: float = 2.0,      # Frequency multiplier per octave
        persistence: float = 0.5,      # Amplitude decay per octave
        noise_strength: float = 0.15,  # Global texture strength
        vegetation_type: str = "auto", # "grass", "tree", "flower", "auto"
    ):
        self.noise_type = noise_type
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.persistence = persistence
        self.noise_strength = noise_strength
        self.vegetation_type = vegetation_type

        # Parameter table based on vegetation type
        self._param_table = {
            "grass": {
                "base_frequency": (0.012, 0.015),  # (x_freq, y_freq)
                "directionality": 0.6,             # Horizontal preference
                "edge_amplification": 1.8,
            },
            "tree": {
                "base_frequency": (0.008, 0.020),
                "directionality": 0.3,
                "edge_amplification": 1.2,
            },
            "flower": {
                "base_frequency": (0.020, 0.020),
                "directionality": 0.0,  # Radial symmetry
                "edge_amplification": 2.0,
            },
            "auto": {
                "base_frequency": (0.010, 0.010),
                "directionality": 0.2,
                "edge_amplification": 1.5,
            },
        }

    def generate_fbm_noise(self, height: int, width: int, seed: int = 42) -> np.ndarray:
        """
        Generate FBM (Fractional Brownian Motion) noise map using the 'noise' library.
        Moving the octave loop to C for a ~10-20x speedup.
        """
        params = self._param_table[self.vegetation_type]
        fx, fy = params["base_frequency"]
        directionality = params["directionality"]
        
        # Select noise function (pnoise2 = Perlin, snoise2 = Simplex)
        if self.noise_type == "perlin":
            noise_fn = noise.pnoise2
        else:
            noise_fn = noise.snoise2

        # Optimization params
        stretch = 1.0 + directionality
        offset_y = 0.1 * directionality
        
        # Build noise map
        # Although we still loop in Python over pixels, the fBm calculation (octaves) 
        # is now entirely in C.
        noise_map = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Apply scaling and custom directionality
                nx = x * fx * stretch
                ny = y * fy + offset_y
                
                # base determines the seed
                noise_map[y, x] = noise_fn(
                    nx, ny,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    base=seed
                )

        # The 'noise' library returns values roughly in range [-1, 1].
        # We normalize to [0, 1] for texture injection.
        noise_min = noise_map.min()
        noise_max = noise_map.max()
        if noise_max > noise_min:
            noise_map = (noise_map - noise_min) / (noise_max - noise_min)
            
        return noise_map

    def generate_vegetation_texture(
        self,
        image: np.ndarray,        # RGB [0, 255]
        vegetation_mask: np.ndarray,  # Binary [H, W]
        seed: int = None,
    ) -> np.ndarray:
        """
        Inject synthesized texture into vegetation regions.
        """
        if seed is None:
            seed = int(np.random.randint(0, 2**31))

        H, W = image.shape[:2]

        # Step 1: Generate FBM Noise Map
        noise_map = self.generate_fbm_noise(H, W, seed=seed)

        # Step 2: Edge-based local amplification
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edge_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_strength = np.sqrt(edge_x**2 + edge_y**2)
        edge_strength = cv2.normalize(edge_strength, None, 0, 1, cv2.NORM_MINMAX)

        params = self._param_table[self.vegetation_type]
        edge_amp = params["edge_amplification"]

        local_strength = self.noise_strength * (1.0 + edge_amp * edge_strength)
        local_strength = np.clip(local_strength, 0, 0.5)

        # Step 3: Apply mask and inject texture
        enhanced = image.copy().astype(np.float32) / 255.0

        veg_mask_float = vegetation_mask.astype(np.float32)
        texture = noise_map * local_strength * veg_mask_float

        # Colorize texture: G channel gets more weight for vegetation
        enhanced[..., 0] += texture * 0.8  # R
        enhanced[..., 1] += texture * 1.2  # G
        enhanced[..., 2] += texture * 0.8  # B

        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        return enhanced

class VegetationGuidedSharpener:
    """
    Guided Sharpness: Transfer edge structures from HR reference to SR output in vegetation regions.
    """

    def __init__(
        self,
        radius: int = 5,
        epsilon: float = 1e-3,
        strength: float = 1.0,
        edge_threshold: float = 0.05,
        mask_blur: int = 3,
    ):
        self.radius = radius
        self.epsilon = epsilon
        self.strength = strength
        self.edge_threshold = edge_threshold
        self.mask_blur = mask_blur

    def guided_filter(
        self,
        guide: np.ndarray,
        input_img: np.ndarray,
        radius: int,
        epsilon: float,
    ) -> np.ndarray:
        """Guided Filter implementation."""
        assert guide.shape == input_img.shape

        if len(guide.shape) == 3:
            guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            guide_gray = guide.astype(np.float32)

        input_f = input_img.astype(np.float32)
        win_size = (radius * 2 + 1, radius * 2 + 1)

        mean_I = cv2.boxFilter(guide_gray, -1, win_size, normalize=True)
        mean_p = cv2.boxFilter(input_f, -1, win_size, normalize=True)

        mean_I_sq = cv2.boxFilter(guide_gray ** 2, -1, win_size, normalize=True)
        var_I = mean_I_sq - mean_I ** 2

        if len(input_f.shape) == 3:
            mean_Ip = np.zeros_like(mean_p)
            for c in range(input_f.shape[2]):
                mean_Ip[..., c] = cv2.boxFilter(
                    guide_gray * input_f[..., c],
                    -1, win_size, normalize=True
                )
        else:
            mean_Ip = cv2.boxFilter(guide_gray * input_f, -1, win_size, normalize=True)

        cov_Ip = mean_Ip - mean_I[..., np.newaxis] * mean_p if len(input_f.shape) == 3 else mean_Ip - mean_I * mean_p
        
        if len(input_f.shape) == 3:
            a = cov_Ip / (var_I[..., np.newaxis] + epsilon)
            b = mean_p - a * mean_I[..., np.newaxis]
        else:
            a = cov_Ip / (var_I + epsilon)
            b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, win_size, normalize=True)
        mean_b = cv2.boxFilter(b, -1, win_size, normalize=True)

        if len(input_f.shape) == 3:
            output = mean_a * guide_gray[..., np.newaxis] + mean_b
        else:
            output = mean_a * guide_gray + mean_b

        return output

    def compute_edge_strength(self, image: np.ndarray) -> np.ndarray:
        """Compute edge strength map for adaptive sharpening."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        edge_x = sobel(gray, axis=1, mode='constant')
        edge_y = sobel(gray, axis=0, mode='constant')
        magnitude = np.sqrt(edge_x**2 + edge_y**2)

        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        edge_strength = np.tanh((magnitude - self.edge_threshold) / 0.05)
        edge_strength = np.clip(edge_strength, 0, 1)
        return edge_strength

    def sharpen_vegetation_region(
        self,
        hr_image: np.ndarray,      # Reference HR image [H, W, 3]
        sr_image: np.ndarray,      # Target SR image [H, W, 3]
        vegetation_mask: np.ndarray, # Binary mask [H, W]
    ) -> np.ndarray:
        """Main Guided Sharpness process."""
        H, W = sr_image.shape[:2]
        
        # Resize HR if needed
        if hr_image.shape[:2] != (H, W):
            hr_reference = cv2.resize(hr_image, (W, H), interpolation=cv2.INTER_LANCZOS4)
        else:
            hr_reference = hr_image

        sr_f = sr_image.astype(np.float32) / 255.0
        hr_f = hr_reference.astype(np.float32) / 255.0

        # Step 1: Smooth mask
        veg_mask_f = vegetation_mask.astype(np.float32)
        veg_mask_smooth = cv2.GaussianBlur(
            veg_mask_f, (self.mask_blur*2+1, self.mask_blur*2+1), sigmaX=1
        )

        # Step 2: Edge strength
        edge_strength = self.compute_edge_strength(sr_image)
        edge_in_veg = edge_strength * veg_mask_smooth

        # Step 3: Guided Filter
        guided_sharp = self.guided_filter(
            guide=hr_f,
            input_img=sr_f,
            radius=self.radius,
            epsilon=self.epsilon,
        )

        # Step 4: Detail extraction
        detail = guided_sharp - sr_f

        # Step 5: Adaptive strength adjustment
        adaptive_strength = self.strength * (0.2 + 0.8 * edge_in_veg[..., np.newaxis])
        sharpened = sr_f + detail * adaptive_strength

        # Step 6: Blend back
        mask_3ch = veg_mask_smooth[..., np.newaxis]
        output = sharpened * mask_3ch + sr_f * (1 - mask_3ch)

        return np.clip(output * 255, 0, 255).astype(np.uint8)

def apply_vegetation_post_processing(
    sr_output: np.ndarray,
    vegetation_mask: np.ndarray,
    method: str = "guided_sharpness",
    hr_image: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    Apply selected post-processing for vegetation areas.
    """
    if vegetation_mask is None or vegetation_mask.sum() == 0:
        return sr_output

    if method == "noise_synthesis":
        synthesizer = VegetationNoiseSynthesizer(
            noise_type=kwargs.get('noise_type', 'simplex'),
            vegetation_type=kwargs.get('vegetation_type', 'auto'),
            noise_strength=kwargs.get('strength', 0.15),
        )
        return synthesizer.generate_vegetation_texture(sr_output, vegetation_mask)
    
    elif method == "guided_sharpness":
        if hr_image is None:
            print("Warning: Guided Sharpness requires HR reference image. Skipping.")
            return sr_output
            
        sharpener = VegetationGuidedSharpener(
            radius=kwargs.get('radius', 5),
            strength=kwargs.get('strength', 0.8),
        )
        return sharpener.sharpen_vegetation_region(
            hr_image=hr_image,
            sr_image=sr_output,
            vegetation_mask=vegetation_mask
        )
    
    return sr_output
