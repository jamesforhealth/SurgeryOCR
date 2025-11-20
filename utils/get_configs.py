from pathlib import Path
import json
from typing import Dict, Any, List, Optional

def _get_json_data(path: Path, error_msg: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(error_msg)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_roi_config(path: Path = Path("config/rois.json"), video_name: Optional[str] = None) -> dict[str, List[int]]:
    """
    加载ROI配置，支持根据视频名称自动选择对应机型的座标
    
    Args:
        path: ROI配置文件路径（默认: config/rois.json）
        video_name: 视频文件名（不含扩展名），用于查找对应的机型
    
    Returns:
        ROI配置字典 {region_name: [x1, y1, x2, y2]}
    
    配置文件结构:
        {
          "video_machine_mapping": {
            "video1": 1,  // 机型编号
            "video2": 2
          },
          "regions": {
            "region1": {
              "1": [[roi_coords], [header_coords]],
              "2": [[roi_coords], [header_coords]]
            }
          }
        }
    
    使用示例:
        # 使用默认配置（机型1）
        roi_dict = load_roi_config()
        
        # 根据视频自动选择机型
        roi_dict = load_roi_config(video_name="2024-10-18周建碧OS")
    """
    data = _get_json_data(path, f"ROI config not found: {path}")
    
    # 确定要使用的机型编号
    machine_id = "1"  # 默认使用机型1
    
    if video_name and "video_machine_mapping" in data:
        mapping = data["video_machine_mapping"]
        if video_name in mapping:
            machine_id = str(mapping[video_name])
            print(f"根据视频 '{video_name}' 自动选择机型 {machine_id}")
    
    # 处理新格式（包含 regions 字段）
    if "regions" in data:
        regions_data = data["regions"]
        result = {}
        
        for region_name, machine_configs in regions_data.items():
            if region_name.startswith('_'):  # 跳过注释字段
                continue
            
            if isinstance(machine_configs, dict):
                # 新格式：{"1": [[coords], [header]], "2": [[coords], [header]]}
                if machine_id in machine_configs:
                    region_data = machine_configs[machine_id]
                    if isinstance(region_data, list) and len(region_data) >= 1:
                        # 取第一个元素作为 ROI 座标
                        result[region_name] = region_data[0]
                else:
                    print(f"警告: 区域 '{region_name}' 没有机型 {machine_id} 的配置，跳过")
            else:
                print(f"警告: 区域 '{region_name}' 配置格式错误，跳过")
        
        return result
    
    # 兼容旧格式（直接在顶层定义 regions）
    result = {}
    for region_name, region_data in data.items():
        if region_name.startswith('_') or region_name in ['video_machine_mapping', 'regions']:
            continue
            
        if isinstance(region_data, list):
            if len(region_data) == 4:
                # 旧格式: 直接座标 [x1, y1, x2, y2]
                result[region_name] = region_data
            elif len(region_data) == 2 and isinstance(region_data[0], list):
                # 旧格式: [[roi_coords], [header_coords]]
                result[region_name] = region_data[0]
    
    return result

def load_roi_header_config(path: Path = Path("config/rois.json"), video_name: Optional[str] = None) -> dict[str, List[int]]:
    """
    加载ROI header配置，支持根据视频名称自动选择对应机型的座标
    
    Args:
        path: ROI配置文件路径（默认: config/rois.json）
        video_name: 视频文件名（不含扩展名），用于查找对应的机型
    
    Returns:
        ROI header配置字典 {region_name: [x1, y1, x2, y2]}
    """
    data = _get_json_data(path, f"ROI config not found: {path}")
    
    # 确定要使用的机型编号
    machine_id = "1"  # 默认使用机型1
    
    if video_name and "video_machine_mapping" in data:
        mapping = data["video_machine_mapping"]
        if video_name in mapping:
            machine_id = str(mapping[video_name])
    
    # 处理新格式（包含 regions 字段）
    if "regions" in data:
        regions_data = data["regions"]
        result = {}
        
        for region_name, machine_configs in regions_data.items():
            if region_name.startswith('_'):  # 跳过注释字段
                continue
            
            if isinstance(machine_configs, dict):
                # 新格式：{"1": [[coords], [header]], "2": [[coords], [header]]}
                if machine_id in machine_configs:
                    region_data = machine_configs[machine_id]
                    if isinstance(region_data, list) and len(region_data) >= 2:
                        # 取第二个元素作为 header 座标
                        if isinstance(region_data[1], list) and len(region_data[1]) > 0:
                            result[region_name] = region_data[1]
        
        return result
    
    # 兼容旧格式
    result = {}
    for region_name, region_data in data.items():
        if region_name.startswith('_') or region_name in ['video_machine_mapping', 'regions']:
            continue
            
        if isinstance(region_data, list) and len(region_data) == 2 and isinstance(region_data[0], list):
            # 旧格式: [[roi_coords], [header_coords]]
            if isinstance(region_data[1], list) and len(region_data[1]) > 0:
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

def update_video_machine_mapping(video_name: str, machine_id: int, path: Path = Path("config/rois.json")) -> bool:
    """
    更新視頻到機型的映射關係
    
    Args:
        video_name: 視頻文件名（不含扩展名）
        machine_id: 機型編號（1 或 2）
        path: ROI配置文件路徑（默認: config/rois.json）
    
    Returns:
        bool: 更新成功返回True，失敗返回False
    
    使用示例:
        # 將視頻標記為機型1
        update_video_machine_mapping("2024-10-18周建碧OS", 1)
        
        # 將視頻標記為機型2
        update_video_machine_mapping("2024-10-04蔡淑淡os", 2)
    """
    try:
        # 讀取現有配置
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # 如果檔案不存在，創建基本結構
            data = {
                "_comment": "整合的ROI配置文件，包含视频-机型映射和各机型的ROI座标",
                "video_machine_mapping": {
                    "_comment": "视频名称(不含扩展名)到机型编号的映射，1=机型A，2=机型B"
                },
                "regions": {}
            }
        
        # 確保 video_machine_mapping 欄位存在
        if "video_machine_mapping" not in data:
            data["video_machine_mapping"] = {
                "_comment": "视频名称(不含扩展名)到机型编号的映射，1=机型A，2=机型B"
            }
        
        # 更新映射
        data["video_machine_mapping"][video_name] = machine_id
        
        # 寫回文件
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 已將視頻 '{video_name}' 映射到機型 {machine_id}")
        return True
        
    except Exception as e:
        print(f"✗ 更新視頻機型映射失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_video_machine_id(video_name: str, path: Path = Path("config/rois.json")) -> Optional[int]:
    """
    獲取視頻對應的機型編號
    
    Args:
        video_name: 視頻文件名（不含扩展名）
        path: ROI配置文件路徑（默認: config/rois.json）
    
    Returns:
        int: 機型編號（1 或 2），如果未配置則返回None
    """
    try:
        if not path.exists():
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "video_machine_mapping" in data:
            return data["video_machine_mapping"].get(video_name)
        
        return None
    except Exception as e:
        print(f"獲取視頻機型編號失敗: {e}")
        return None


def read_surgery_stage_rois(path: Path) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)