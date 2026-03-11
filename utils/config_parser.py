"""
Configuration file parser
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default config
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve paths relative to config file directory
    config_dir = config_path.parent
    config = resolve_paths(config, config_dir)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default config dictionary
    """
    return {
        'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        
        # Model paths
        'models': {
            'speciesnet': 'models/speciesnet.pth',
            'sam': 'models/sam_vit_h.pth',
            'bisenet': 'models/bisenet_face_parsing.pth',
            'person_detector': 'models/yolov8_person.pt',
        },
        
        # SAM settings
        'sam': {
            'model_type': 'vit_h',
            'points_per_side': 32,
            'pred_iou_thresh': 0.88,
            'stability_score_thresh': 0.95,
        },
        
        # SpeciesNet settings
        'speciesnet': {
            'confidence_threshold': 0.5,
            'furry_classes': ['cat', 'dog', 'horse', 'cow', 'sheep', 'rabbit', 'bear', 'fox', 'wolf'],
        },
        
        # BiSeNet settings
        'bisenet': {
            'num_classes': 19,
            'face_parsing': True,
        },
        
        # Mask merging settings
        'merging': {
            'gaussian_sigma': 3.0,
            'edge_blur_radius': 5,
        },
        
        # Processing settings
        'processing': {
            'max_image_size': 2048,
            'batch_size': 1,
        },
        
        # Output settings
        'output': {
            'save_masks': True,
            'save_visualization': True,
            'output_dir': 'output',
        }
    }


def resolve_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """
    Resolve relative paths in config to absolute paths
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths
        
    Returns:
        Config with resolved paths
    """
    if 'models' in config:
        for key, path in config['models'].items():
            if isinstance(path, str) and not Path(path).is_absolute():
                config['models'][key] = str(base_dir / path)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class Config:
    """
    Configuration manager with dot notation access
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()
