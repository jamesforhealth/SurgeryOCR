import json
import os
from pathlib import Path
import argparse
import tempfile # 用於安全覆寫
import shutil   # 用於移動文件

def fill_missing_frames(target_filepath: Path):
    """
    讀取一個只包含變化幀的 JSONL 文件 (例如 region2.jsonl)，
    填充缺失的幀，使其包含從最小到最大幀號的所有連續幀記錄，
    並直接覆寫原始文件。
    """
    if not target_filepath.exists():
        print(f"錯誤：目標文件不存在 {target_filepath}")
        return

    records_by_frame = {}
    video_name = None
    min_frame = float('inf')
    max_frame = float('-inf')

    print(f"正在讀取目標文件：{target_filepath}")
    try:
        with open(target_filepath, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    image_path_str = record.get("images", "")
                    # response = record.get("response", "") # response 在填充時才需要

                    # 從 image_path 解析幀號和影片名
                    image_path = Path(image_path_str)
                    filename = image_path.name # e.g., "frame_1910.png"
                    if filename.startswith("frame_") and filename.endswith(".png"):
                        frame_idx_str = filename[len("frame_"):-len(".png")]
                        frame_idx = int(frame_idx_str)

                        # 嘗試從路徑提取影片名 (只需要一次)
                        if video_name is None and len(image_path.parts) > 1:
                            # 預期路徑格式: "VIDEO_NAME/region2/frame_..." 或 "region2/frame_..."
                            # 如果路徑包含影片名，則取第一個部分
                            if len(image_path.parts) > 2:
                                video_name = image_path.parts[0]
                            else:
                                # 如果路徑不直接包含影片名，嘗試從文件路徑推斷
                                video_name = target_filepath.parent.name
                            print(f"檢測到影片名：{video_name}")


                        records_by_frame[frame_idx] = record # 儲存原始記錄
                        min_frame = min(min_frame, frame_idx)
                        max_frame = max(max_frame, frame_idx)
                    else:
                        print(f"警告：第 {line_num + 1} 行無法解析幀號: {image_path_str}")

                except json.JSONDecodeError:
                    print(f"警告：第 {line_num + 1} 行 JSON 格式錯誤，已跳過: {line}")
                except (ValueError, KeyError, AttributeError) as e:
                    print(f"警告：處理第 {line_num + 1} 行時出錯 ({e})，已跳過: {line}")

    except IOError as e:
        print(f"錯誤：讀取文件時出錯 {target_filepath}: {e}")
        return
    except Exception as e:
        print(f"讀取文件時發生未知錯誤: {e}")
        return

    if not records_by_frame:
        print("錯誤：未能從輸入文件中讀取有效記錄。")
        return
    if video_name is None:
        # 如果無法從 image path 推斷，再次嘗試從文件路徑獲取
        video_name = target_filepath.parent.name
        if not video_name:
             print("錯誤：無法確定影片名稱，無法繼續。")
             return
        print(f"從文件路徑推斷影片名：{video_name}")


    print(f"原始記錄範圍：幀 {min_frame} 到 {max_frame}")
    print(f"總共 {len(records_by_frame)} 筆有效原始記錄。")

    filled_records = []
    last_known_response = "" # 初始值

    print(f"正在填充從 {min_frame} 到 {max_frame} 的所有幀...")
    # 確保按幀號順序處理
    sorted_original_frames = sorted(records_by_frame.keys())
    original_record_idx = 0

    for current_frame in range(min_frame, max_frame + 1):
        if original_record_idx < len(sorted_original_frames) and sorted_original_frames[original_record_idx] == current_frame:
            # 使用原始記錄
            record = records_by_frame[current_frame]
            last_known_response = record.get("response", "") # 更新最後已知的 response
            filled_records.append(record)
            original_record_idx += 1
        else:
            # 填充缺失的幀
            # 確保 image path 使用正確的 video_name
            image_path = f"{video_name}/region2/frame_{current_frame}.png"
            new_record = {
                "query": "<image>",
                "response": last_known_response, # 使用上一個已知幀的 response
                "images": image_path
            }
            filled_records.append(new_record)

    print(f"填充完成，總共 {len(filled_records)} 筆記錄。")

    # --- 安全地覆寫原始文件 ---
    # 1. 寫入臨時文件
    temp_file = None
    try:
        # 在與目標文件相同的目錄下創建臨時文件，以確保 rename 操作在同一個文件系統
        temp_fd, temp_path_str = tempfile.mkstemp(dir=target_filepath.parent, prefix=target_filepath.stem + '_', suffix='.tmp')
        temp_file = Path(temp_path_str)

        print(f"正在寫入臨時文件：{temp_file}")
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as outfile:
            for record in filled_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. 用臨時文件替換原始文件
        print(f"正在用臨時文件覆寫目標文件：{target_filepath}")
        shutil.move(str(temp_file), str(target_filepath)) # 使用 shutil.move 進行原子性替換 (在多數系統上)
        print("文件覆寫成功。")

    except IOError as e:
        print(f"錯誤：寫入臨時文件或覆寫文件時出錯: {e}")
        # 如果出錯，嘗試刪除可能遺留的臨時文件
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                print(f"已刪除臨時文件：{temp_file}")
            except OSError as unlink_e:
                print(f"錯誤：刪除臨時文件失敗: {unlink_e}")
    except Exception as e:
        print(f"處理文件時發生未知錯誤: {e}")
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                print(f"已刪除臨時文件：{temp_file}")
            except OSError as unlink_e:
                print(f"錯誤：刪除臨時文件失敗: {unlink_e}")
    finally:
        # 確保即使成功移動，路徑變數也不會指向已移動的文件
        if temp_file and temp_file.exists():
             try:
                 temp_file.unlink()
             except OSError:
                 pass # 可能已被移動，忽略錯誤


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="填充 JSONL 文件中缺失的視頻幀記錄，並覆寫原始文件。")
    parser.add_argument("--video_folder_name", type=str, help="包含 region2.jsonl 的影片資料夾名稱 (例如 '2024-12-04-lin')")
    parser.add_argument("--base_dir", type=Path, default=Path("./"), help="包含影片資料夾的基礎目錄 (預設: 'data')")
    parser.add_argument("--region_name", type=str, default="region2", help="要處理的區域名稱 (預設: 'region2')")

    args = parser.parse_args()

    filename = args.region_name + ".jsonl"
    # --- 構建目標文件路徑 ---
    target_file = args.base_dir / args.video_folder_name / filename

    # --- 執行填充函數 ---
    fill_missing_frames(target_file)