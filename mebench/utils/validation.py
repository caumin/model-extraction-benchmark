"""Global learning rate validation to enforce contract compliance."""

import logging
from typing import Dict, Any, List, Optional
import warnings


class LearningRateValidator:
    """Validates learning rates to ensure compliance with global contract.
    
    According to AGENTS.md and the global benchmark contract:
    - Default substitute learning rate MUST be 0.01 across all experiments
    - This ensures fair comparison between attacks
    - Deviations should only be allowed with explicit justification
    """
    
    # Global contract LR
    CONTRACT_LR = 0.01
    
    # Attacks that may legitimately use different LRs with justification
    EXCEPTIONS = {
        "dfme": {
            "student_lr": 0.01,  # Fixed by our implementation
            "generator_lr": 5e-4,  # Generator LR - legitimately different
            "justification": "Generator uses Adam with different dynamics than substitute SGD"
        },
        "maze": {
            "clone_lr": 0.01,  # Clone model LR
            "generator_lr": 1e-4,  # Generator LR - legitimately different
            "justification": "Generator uses Adam for GAN training, clone uses SGD for classifier"
        },
        "game": {
            "substitute_lr": 0.01,  # Must follow contract
            "generator_lr": 2e-4,  # Generator LR - legitimately different
            "justification": "Generator uses Adam for GAN training"
        },
        "dfms": {
            "substitute_lr": 0.01,  # Must follow contract
            "generator_lr": 1e-4,  # Generator LR - legitimately different
            "justification": "Generator uses Adam for GAN training"
        },
        "es_attack": {
            "substitute_lr": 0.01,  # Must follow contract
            "justification": "Uses standard classifier training"
        },
        "knockoff_nets": {
            "policy_lr": 0.01,  # Policy learning can use different LR
            "justification": "Bandit policy learning has different dynamics"
        }
    }
    
    def __init__(self, strict_mode: bool = True):
        """Initialize LR validator.
        
        Args:
            strict_mode: If True, raises exceptions for violations.
                       If False, issues warnings.
        """
        self.strict_mode = strict_mode
        self.violations = []
        
    def validate_attack_config(
        self, 
        attack_name: str, 
        config: Dict[str, Any]
    ) -> List[str]:
        """Validate learning rates in attack configuration.
        
        Args:
            attack_name: Name of the attack (e.g., "dfme", "activethief")
            config: Attack configuration dictionary
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for various LR parameter names
        lr_params = self._extract_lr_params(config)
        
        for param_name, lr_value in lr_params.items():
            violation = self._check_lr_compliance(
                attack_name, param_name, lr_value
            )
            if violation:
                errors.append(violation)
                self.violations.append(violation)
        
        return errors
    
    def _extract_lr_params(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Extract all learning rate parameters from config."""
        lr_params = {}
        
        # Common LR parameter names
        lr_keys = [
            "lr", "learning_rate", "substitute_lr", "student_lr",
            "generator_lr", "clone_lr", "policy_lr", "attack_lr",
            "base_lr", "init_lr"
        ]
        
        for key in lr_keys:
            if key in config:
                try:
                    lr_value = float(config[key])
                    lr_params[key] = lr_value
                except (ValueError, TypeError):
                    pass
        
        # Check nested structures
        if "optimizer" in config:
            opt_config = config["optimizer"]
            if isinstance(opt_config, dict) and "lr" in opt_config:
                try:
                    lr_value = float(opt_config["lr"])
                    lr_params["optimizer.lr"] = lr_value
                except (ValueError, TypeError):
                    pass
        
        if "substitute" in config:
            sub_config = config["substitute"]
            if isinstance(sub_config, dict):
                if "optimizer" in sub_config and "lr" in sub_config["optimizer"]:
                    try:
                        lr_value = float(sub_config["optimizer"]["lr"])
                        lr_params["substitute.optimizer.lr"] = lr_value
                    except (ValueError, TypeError):
                        pass
        
        return lr_params
    
    def _check_lr_compliance(
        self, 
        attack_name: str, 
        param_name: str, 
        lr_value: float
    ) -> Optional[str]:
        """Check if a specific LR parameter complies with contract."""
        
        # Normalize attack name
        attack_key = attack_name.lower().replace("-", "_").replace(" ", "_")
        
        # Check if this attack has specific exceptions
        if attack_key in self.EXCEPTIONS:
            exception_info = self.EXCEPTIONS[attack_key]
            
            # Check if this parameter is in the exception
            if param_name in exception_info:
                expected_lr = exception_info[param_name]
                tolerance = 1e-6  # Allow for floating point precision
                
                if abs(lr_value - expected_lr) > tolerance:
                    return (f"Attack '{attack_name}' parameter '{param_name}' has LR {lr_value}, "
                           f"but expected {expected_lr} per contract exception. "
                           f"Justification: {exception_info['justification']}")
                else:
                    return None  # Exception applies and value is correct
            else:
                # Parameter not in exception - must use contract LR
                if abs(lr_value - self.CONTRACT_LR) > 1e-6:
                    return (f"Attack '{attack_name}' parameter '{param_name}' has LR {lr_value}, "
                           f"but must use contract LR {self.CONTRACT_LR}. "
                           f"This parameter is not covered by attack-specific exceptions.")
        
        # Default: must use contract LR
        if abs(lr_value - self.CONTRACT_LR) > 1e-6:
            return (f"Attack '{attack_name}' parameter '{param_name}' has LR {lr_value}, "
                   f"but must use contract LR {self.CONTRACT_LR}. "
                   f"All substitute classifier training must use lr={self.CONTRACT_LR}.")
        
        return None
    
    def validate_all_attacks(
        self, 
        attack_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Validate multiple attack configurations.
        
        Args:
            attack_configs: Dictionary mapping attack names to their configs
            
        Returns:
            Dictionary mapping attack names to list of violations
        """
        all_violations = {}
        
        for attack_name, config in attack_configs.items():
            violations = self.validate_attack_config(attack_name, config)
            if violations:
                all_violations[attack_name] = violations
        
        return all_violations
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get summary of all violations found."""
        return {
            "total_violations": len(self.violations),
            "contract_lr": self.CONTRACT_LR,
            "violations": self.violations,
            "strict_mode": self.strict_mode
        }
    
    def log_violations(self, violations: List[str], attack_name: str) -> None:
        """Log violations based on strict mode."""
        for violation in violations:
            if self.strict_mode:
                logging.error(f"LR Validation Error [{attack_name}]: {violation}")
            else:
                logging.warning(f"LR Validation Warning [{attack_name}]: {violation}")
    
    def auto_fix_config(
        self, 
        attack_name: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Automatically fix common LR violations in config.
        
        Args:
            attack_name: Name of the attack
            config: Original configuration
            
        Returns:
            Fixed configuration with corrected LRs
        """
        fixed_config = config.copy()
        
        # Normalize attack name
        attack_key = attack_name.lower().replace("-", "_").replace(" ", "_")
        
        # Get expected LRs for this attack
        if attack_key in self.EXCEPTIONS:
            exception_info = self.EXCEPTIONS[attack_key]
            
            # Apply exception-based fixes
            for param_name, expected_lr in exception_info.items():
                if param_name == "justification":
                    continue
                    
                # Fix various config locations
                if param_name in fixed_config:
                    if abs(float(fixed_config[param_name]) - expected_lr) > 1e-6:
                        logging.info(f"Auto-fixing {attack_name}.{param_name}: "
                                   f"{fixed_config[param_name]} → {expected_lr}")
                        fixed_config[param_name] = expected_lr
                
                # Check nested optimizer configs
                if param_name.endswith("_lr") and "optimizer" in fixed_config:
                    opt_config = fixed_config["optimizer"].copy()
                    if "lr" in opt_config:
                        if abs(float(opt_config["lr"]) - expected_lr) > 1e-6:
                            logging.info(f"Auto-fixing {attack_name}.optimizer.lr: "
                                       f"{opt_config['lr']} → {expected_lr}")
                            opt_config["lr"] = expected_lr
                            fixed_config["optimizer"] = opt_config
        
        else:
            # Default: ensure all LR params use contract LR
            lr_params = self._extract_lr_params(fixed_config)
            for param_name, lr_value in lr_params.items():
                if abs(lr_value - self.CONTRACT_LR) > 1e-6:
                    # Fix nested configs
                    if "." in param_name:
                        keys = param_name.split(".")
                        current = fixed_config
                        for key in keys[:-1]:
                            if key not in current:
                                current[key] = {}
                            current = current[key]
                        current[keys[-1]] = self.CONTRACT_LR
                    else:
                        fixed_config[param_name] = self.CONTRACT_LR
                    
                    logging.info(f"Auto-fixing {attack_name}.{param_name}: "
                               f"{lr_value} → {self.CONTRACT_LR}")
        
        return fixed_config


# Global validator instance
_global_validator = LearningRateValidator(strict_mode=False)


def validate_learning_rates(
    attack_name: str, 
    config: Dict[str, Any], 
    strict_mode: bool = False
) -> List[str]:
    """Convenience function to validate learning rates in attack config.
    
    Args:
        attack_name: Name of the attack
        config: Attack configuration
        strict_mode: Whether to raise exceptions vs warnings
        
    Returns:
        List of validation violations
    """
    validator = LearningRateValidator(strict_mode=strict_mode)
    violations = validator.validate_attack_config(attack_name, config)
    
    if violations:
        validator.log_violations(violations, attack_name)
        if strict_mode:
            raise ValueError(f"Learning rate validation failed for {attack_name}: {violations}")
    
    return violations


def auto_fix_learning_rates(
    attack_name: str, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Convenience function to auto-fix LR violations in attack config.
    
    Args:
        attack_name: Name of the attack
        config: Original configuration
        
    Returns:
        Fixed configuration
    """
    validator = LearningRateValidator(strict_mode=False)
    return validator.auto_fix_config(attack_name, config)