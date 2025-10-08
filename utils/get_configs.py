from pathlib import Path
import json
from typing import Dict, Any, List


def _get_json_data(path: Path, error_msg: str) -> dict[str, List[int]]:
    if not path.exists():
        raise FileNotFoundError(error_msg)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_roi_config(path: Path) -> dict[str, List[int]]:
    """Load ROI dictionary from JSON file. 
    Supports both old format (direct coordinates) and new format (with header coordinates).
    Returns only the main ROI coordinates.
    """
    data = _get_json_data(path, f"ROI config not found: {path}")
    
    # Handle both old and new format
    result = {}
    for region_name, region_data in data.items():
        if isinstance(region_data, list):
            if len(region_data) == 4:
                # Old format: direct coordinates [x1, y1, x2, y2]
                result[region_name] = region_data
            elif len(region_data) == 2 and isinstance(region_data[0], list):
                # New format: [[roi_coords], [header_coords]] - take first group
                result[region_name] = region_data[0]
            else:
                raise ValueError(f"Invalid ROI data format for region '{region_name}': {region_data}")
        else:
            raise ValueError(f"Invalid ROI data type for region '{region_name}': {type(region_data)}")
    
    return result

def load_roi_header_config(path: Path) -> dict[str, List[int]]:
    """Load ROI header coordinates from JSON file.
    Returns header coordinates for regions that have them, empty dict if none.
    """
    data = _get_json_data(path, f"ROI config not found: {path}")
    
    result = {}
    for region_name, region_data in data.items():
        if isinstance(region_data, list) and len(region_data) == 2 and isinstance(region_data[0], list):
            # New format: [[roi_coords], [header_coords]] - take second group
            result[region_name] = region_data[1]
    
    return result

def load_stage_config(path: Path) -> dict[str, List[int]]:
    """Load OCR activation stages dictionary from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Stage config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, List[int]] = json.load(f)
    return data

def load_ocr_char_sets_config(path: Path) -> dict[str, str]:
    """Load OCR character sets dictionary from JSON file."""
    if not path.exists():
        return {}  # Return empty dict if file doesn't exist, as it's optional
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)
    return data

def load_pattern_name_mapping(path: Path) -> Dict[str, Dict[str, str]]:
    """Load pattern ID to name mapping from JSON file."""
    if not path.exists():
        print(f"Pattern name mapping file not found, skipping: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, str]] = json.load(f)
    return data

def get_diff_rule_config_path() -> Path:
    """回傳 config/diff_rule.json 路徑"""
    return Path("config") / "diff_rule.json"

def load_diff_rules() -> Dict[str, Any]:
    """載入差異分析規則配置"""
    config_path = get_diff_rule_config_path()
    
    # 預設配置
    default_config = {
        "PEDAL": {
            "analysis_mode": "sub_roi",
            "sub_roi_coords": [20, 13, 26, 19],
            "diff": "rmse",
            "diff_threshold": 30.0,
            "description": "PEDAL區域精細分析範圍"
        },
        "STAGE": {
            "analysis_mode": "full_roi",
            "diff": "rmse",
            "diff_threshold": 15.0,
            "description": "STAGE區域使用完整ROI"
        }
    }
    
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # 與預設配置合併，確保缺失的鍵有預設值
                for region in default_config:
                    if region not in config:
                        config[region] = default_config[region]
                    else:
                        for key in default_config[region]:
                            if key not in config[region]:
                                config[region][key] = default_config[region][key]
                return config
        else:
            print(f"配置檔案不存在，使用預設配置: {config_path}")
            return default_config
    except Exception as e:
        print(f"載入差異規則配置失敗: {e}，使用預設配置")
        return default_config

def save_diff_rules(config: Dict[str, Any]) -> bool:
    """儲存差異分析規則配置"""
    config_path = get_diff_rule_config_path()
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"儲存差異規則配置失敗: {e}")
        return False

def load_setting_regions_config(path: Path = None) -> Dict[str, Any]:
    """載入設定值區域配置"""
    if path is None:
        path = Path("config") / "setting_regions.json"
    
    # 預設配置
    default_config = {
        "regions_with_setting_detection": ["region1", "region2", "region3", "region4", "region5"],
        "detection_config": {
            "sub_region_coords": {
                "x1": 60,
                "x2": 120,
                "y1": 46,
                "y2": 50
            },
            "white_pixel_threshold": 0.02,
            "description": "如果子區域白色像素比例超過2%，判定為運作值（大數字）；否則為設定值（小數字）"
        },
        "regions_without_setting_detection": ["region6", "region7"],
        "default_setting_value": False
    }
    
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # 與預設配置合併，確保缺失的鍵有預設值
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        else:
            print(f"設定區域配置檔案不存在，使用預設配置: {path}")
            return default_config
    except Exception as e:
        print(f"載入設定區域配置失敗: {e}，使用預設配置")
        return default_config