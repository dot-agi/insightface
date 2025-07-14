"""
Configuration loader for ArcFace evaluation framework.
Supports YAML configs with hierarchical merging and environment variable substitution.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader with support for YAML configs and environment variable substitution"""
    
    def __init__(self, config_dir: str = "run_configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_path: str, env_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable substitution
        
        Args:
            config_path: Path to config file (relative to config_dir or absolute)
            env_vars: Additional environment variables to use for substitution
            
        Returns:
            Configuration dictionary
        """
        # Resolve config path
        if not os.path.isabs(config_path):
            config_path = self.config_dir / config_path
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        logger.info(f"Loading configuration from: {config_path}")
        
        # Load YAML content
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Substitute environment variables
        content = self._substitute_env_vars(content, env_vars)
        
        # Parse YAML
        config = yaml.safe_load(content)
        
        # Validate basic structure
        self._validate_config(config)
        
        return config
        
    def _substitute_env_vars(self, content: str, additional_vars: Optional[Dict[str, str]] = None) -> str:
        """Substitute environment variables in config content"""
        import re
        
        # Prepare environment variables
        env_vars = dict(os.environ)
        if additional_vars:
            env_vars.update(additional_vars)
            
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) or ""
            
            return env_vars.get(var_name, default_value)
            
        return re.sub(pattern, replace_var, content)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        # Basic validation - can be extended
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Configuration to merge on top
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
        
    def load_evaluation_config(self, config_name: str = "default", **overrides) -> 'EvaluationConfig':
        """
        Load evaluation configuration with overrides
        
        Args:
            config_name: Name of config file (without .yaml extension)
            **overrides: Key-value overrides for config
            
        Returns:
            EvaluationConfig object
        """
        config_path = f"{config_name}.yaml"
        config = self.load_config(config_path)
        
        # Apply overrides
        if overrides:
            override_dict = self._flatten_overrides(overrides)
            config = self.merge_configs(config, override_dict)
            
        return EvaluationConfig(config)
        
    def _flatten_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat overrides to nested dictionary"""
        result = {}
        
        for key, value in overrides.items():
            keys = key.split('.')
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
                
            current[keys[-1]] = value
            
        return result


class EvaluationConfig:
    """Configuration object for evaluation with easy access to nested values"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.network')"""
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
    @property
    def raw(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return self._config
        
    # Convenience properties for common config sections
    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get('model', {})
        
    @property
    def data(self) -> Dict[str, Any]:
        return self._config.get('data', {})
        
    @property
    def evaluation(self) -> Dict[str, Any]:
        return self._config.get('evaluation', {})
        
    @property
    def performance(self) -> Dict[str, Any]:
        return self._config.get('performance', {})
        
    @property
    def monitoring(self) -> Dict[str, Any]:
        return self._config.get('monitoring', {})
        
    @property
    def output(self) -> Dict[str, Any]:
        return self._config.get('output', {})
        
    @property
    def wandb(self) -> Dict[str, Any]:
        return self._config.get('wandb', {})
        
    def __str__(self) -> str:
        return yaml.dump(self._config, default_flow_style=False, indent=2)


# Global config loader instance
config_loader = ConfigLoader()


def load_config(config_name: str = "default", **overrides) -> EvaluationConfig:
    """Convenience function to load evaluation configuration"""
    return config_loader.load_evaluation_config(config_name, **overrides)