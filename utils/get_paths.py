import os
from pathlib import Path

_ENV_ANALYSIS_ROOT = os.environ.get("ANALYSIS_DATA_ROOT")
DEFAULT_ANALYSIS_ROOT = Path(_ENV_ANALYSIS_ROOT or "data")


def resolve_video_analysis_dir(video_path_or_name: Path | str, data_root: Path | None = None) -> Path:
    """
    解析視頻分析目錄，支持多層子目錄結構
    
    優先順序：
    1. 如果傳入的是完整路徑且存在，直接使用
    2. 如果傳入的是視頻文件，查找同級的 <stem>/ 目錄
    3. 查找 data/<子目錄>/<video_name>/ (遞歸掃描)
    4. 回退到 data/<video_name>/
    
    Args:
        video_path_or_name: 視頻路徑或名稱
        data_root: data 根目錄
    
    Returns:
        分析目錄的 Path，如果都不存在則返回預期的路徑（data/<video_name>/）
    """
    if data_root is None:
        data_root = DEFAULT_ANALYSIS_ROOT
    env_override_active = _ENV_ANALYSIS_ROOT is not None

    path_obj = Path(video_path_or_name)
    
    # Case 1: 如果是目錄且存在，直接返回
    if path_obj.is_dir():
        return path_obj
    
    # Case 2: 如果是視頻文件，查找同級的分析目錄
    if path_obj.is_file():
        video_name = path_obj.stem
        if env_override_active:
            override_dir = data_root / video_name
            if override_dir.exists():
                return override_dir
        analysis_dir = path_obj.parent / video_name
        if analysis_dir.exists():
            return analysis_dir
    else:
        video_name = path_obj.stem if path_obj.suffix else str(path_obj)
    
    # Case 3: 遞歸掃描 data_root 下的子目錄
    if data_root.exists():
        for subdir in data_root.rglob(video_name):
            if subdir.is_dir() and subdir.name == video_name:
                # 確認這是分析目錄（包含 stage_analysis.json 或其他分析文件）
                if (subdir / "stage_analysis.json").exists() or \
                   any(subdir.glob("region*")) or \
                   (subdir / "frame_cache").exists():
                    return subdir
    
    # Case 4: 回退到直接在 data_root 下
    direct_path = data_root / video_name
    if direct_path.exists():
        return direct_path
    
    # 如果都不存在，返回預期的直接路徑（用於創建）
    return direct_path
