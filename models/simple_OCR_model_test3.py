import os
import json
import random
import time
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms

# 假設這些是已經定義好的函數和類
from data_loader import load_data_from_jsonl, OCRDataset
from models import PaddleOCRWrapper, TrOCRWrapper, CRNNWrapper
from train import train_with_cross_validation

# 完成多模型 OCR 集成和主函數實現

我會為您完成 `collect_all_validation_predictions` 函數的剩餘部分，並添加必要的主函數來執行這個多模型 OCR 系統。以下是完整的實現：

```python:simple_OCR_model_test3.py
def collect_all_validation_predictions(data_samples, model_dir, output_dir, args):
    """整合所有fold的驗證集預測結果"""
    logger.info("=== 整合所有fold的驗證集預測結果 ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入fold分配
    fold_assignments_path = os.path.join(model_dir, "fold_assignments.json")
    if not os.path.exists(fold_assignments_path):
        logger.error(f"找不到fold分配文件: {fold_assignments_path}")
        return
        
    with open(fold_assignments_path, 'r') as f:
        # 轉換鍵從字符串到整數
        fold_assignments = {int(k): v for k, v in json.load(f).items()}
    
    # 各個模型的完整預測結果
    all_model_predictions = {
        'paddle': {},
        'trocr': {},
        'crnn': {}
    }
    
    # 所有樣本的真實標籤
    ground_truth = {}
    
    # 處理每個樣本
    for idx, (img_path, response, frame_num) in enumerate(tqdm(data_samples, desc="整合預測結果")):
        # 如果樣本被分配到某個fold的驗證集
        if str(idx) in fold_assignments or idx in fold_assignments:
            # 使用字符串或整數鍵
            fold = fold_assignments.get(str(idx), fold_assignments.get(idx))
            fold_dir = os.path.join(model_dir, f"fold_{fold}")
            
            # 保存真實標籤
            ground_truth[frame_num] = response
            
            # 查找每個模型的預測結果
            for model_name in ['paddle', 'trocr', 'crnn']:
                model_results_path = os.path.join(fold_dir, model_name, "val_results.json")
                if os.path.exists(model_results_path):
                    with open(model_results_path, 'r') as f:
                        model_results = json.load(f)
                        
                    # 在預測結果中查找相應frame的預測
                    for pred_info in model_results['predictions']:
                        if pred_info['frame'] == frame_num:
                            all_model_predictions[model_name][frame_num] = pred_info['pred']
                            break
    
    # 計算各模型的準確率
    model_accuracies = {}
    
    for model_name, predictions in all_model_predictions.items():
        if predictions:
            correct = sum(1 for frame in predictions if predictions[frame].strip() == ground_truth.get(frame, "").strip())
            total = len(predictions)
            accuracy = correct / total if total > 0 else 0
            model_accuracies[model_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            logger.info(f"{model_name} 準確率: {accuracy*100:.2f}% ({correct}/{total})")
    
    # 創建比較結果
    comparison_results = []
    
    for frame_num in sorted(ground_truth.keys()):
        true_label = ground_truth[frame_num]
        result = {
            'frame': frame_num,
            'true_label': true_label
        }
        
        # 添加各模型的預測
        for model_name in all_model_predictions:
            pred = all_model_predictions[model_name].get(frame_num, "")
            result[f'{model_name}_pred'] = pred
            result[f'{model_name}_correct'] = (pred.strip() == true_label.strip())
        
        comparison_results.append(result)
    
    # 保存比較結果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # 保存JSON格式
    comparison_json = {
        'model_accuracies': model_accuracies,
        'predictions': comparison_results
    }
    
    with open(os.path.join(output_dir, "model_comparison.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison_json, f, ensure_ascii=False, indent=2)
    
    # 計算模型之間的一致性
    if len(all_model_predictions) >= 2:
        logger.info("\n=== 模型間一致性分析 ===")
        agreement_count = 0
        disagreement_count = 0
        
        for frame in ground_truth:
            # 收集所有可用模型對這個樣本的預測
            frame_preds = [all_model_predictions[m].get(frame, "") for m in all_model_predictions if frame in all_model_predictions[m]]
            
            # 如果有至少兩個模型的預測
            if len(frame_preds) >= 2:
                # 檢查是否所有預測都一樣
                if len(set(frame_preds)) == 1:
                    agreement_count += 1
                else:
                    disagreement_count += 1
        
        total_samples = agreement_count + disagreement_count
        agreement_rate = agreement_count / total_samples if total_samples > 0 else 0
        
        logger.info(f"模型間一致樣本: {agreement_count}/{total_samples} ({agreement_rate*100:.2f}%)")
        logger.info(f"模型間不一致樣本: {disagreement_count}/{total_samples} ({(1-agreement_rate)*100:.2f}%)")
        
        # 找出模型間不一致但有至少一個模型預測正確的樣本
        correct_disagreements = []
        incorrect_agreements = []
        
        for frame in ground_truth:
            true_label = ground_truth[frame]
            model_preds = {m: all_model_predictions[m].get(frame, "") for m in all_model_predictions if frame in all_model_predictions[m]}
            
            # 如果有至少兩個模型的預測
            if len(model_preds) >= 2:
                # 檢查預測是否一致
                if len(set(model_preds.values())) > 1:
                    # 不一致，檢查是否有至少一個正確
                    for model_name, pred in model_preds.items():
                        if pred.strip() == true_label.strip():
                            correct_disagreements.append({
                                'frame': frame,
                                'true_label': true_label,
                                'predictions': model_preds
                            })
                            break
                else:
                    # 一致，檢查是否都錯
                    first_pred = next(iter(model_preds.values()))
                    if first_pred.strip() != true_label.strip():
                        incorrect_agreements.append({
                            'frame': frame,
                            'true_label': true_label,
                            'prediction': first_pred
                        })
        
        # 輸出不一致但至少一個正確的樣本數
        logger.info(f"模型間不一致但至少一個正確的樣本: {len(correct_disagreements)}")
        logger.info(f"模型間一致但全部錯誤的樣本: {len(incorrect_agreements)}")
        
        # 保存這些特殊情況的樣本
        with open(os.path.join(output_dir, "correct_disagreements.json"), 'w', encoding='utf-8') as f:
            json.dump(correct_disagreements, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(output_dir, "incorrect_agreements.json"), 'w', encoding='utf-8') as f:
            json.dump(incorrect_agreements, f, ensure_ascii=False, indent=2)
    
    # 創建模型集成預測
    ensemble_predictions = {}
    
    for frame in ground_truth:
        # 收集所有模型對這個樣本的預測
        frame_preds = {m: all_model_predictions[m].get(frame, "") for m in all_model_predictions if frame in all_model_predictions[m]}
        
        if not frame_preds:
            continue
            
        # 使用加權投票進行集成
        # 權重基於每個模型的整體準確率
        weights = {m: model_accuracies[m]['accuracy'] for m in frame_preds if m in model_accuracies}
        
        # 如果沒有權重信息，使用均等權重
        if not weights:
            weights = {m: 1.0 for m in frame_preds}
        
        # 標準化權重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {m: w/total_weight for m, w in weights.items()}
        
        # 對每個唯一預測計算加權票數
        pred_votes = defaultdict(float)
        for model, pred in frame_preds.items():
            pred_votes[pred.strip()] += weights.get(model, 0)
        
        # 選擇得票最高的預測
        if pred_votes:
            ensemble_pred = max(pred_votes.items(), key=lambda x: x[1])[0]
            ensemble_predictions[frame] = ensemble_pred
    
    # 計算集成模型準確率
    if ensemble_predictions:
        correct = sum(1 for frame in ensemble_predictions if ensemble_predictions[frame] == ground_truth.get(frame, "").strip())
        accuracy = correct / len(ensemble_predictions)
        logger.info(f"集成模型準確率: {accuracy*100:.2f}% ({correct}/{len(ensemble_predictions)})")
        
        # 與最佳單個模型比較
        best_model = max(model_accuracies.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = model_accuracies[best_model]['accuracy']
        
        if accuracy > best_accuracy:
            logger.info(f"集成模型優於最佳單個模型 ({best_model}: {best_accuracy*100:.2f}%)")
        else:
            logger.info(f"最佳單個模型 ({best_model}: {best_accuracy*100:.2f}%) 優於集成模型")
        
        # 保存集成預測結果
        ensemble_results = []
        for frame in sorted(ensemble_predictions.keys()):
            ensemble_results.append({
                'frame': frame,
                'true_label': ground_truth.get(frame, ""),
                'ensemble_pred': ensemble_predictions[frame],
                'is_correct': ensemble_predictions[frame] == ground_truth.get(frame, "").strip()
            })
            
        with open(os.path.join(output_dir, "ensemble_predictions.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy,
                'correct': correct,
                'total': len(ensemble_predictions),
                'predictions': ensemble_results
            }, f, ensure_ascii=False, indent=2)
    
    # 可視化比較結果
    visualize_model_comparison(comparison_df, output_dir)
    
    return {
        'model_accuracies': model_accuracies,
        'ensemble_accuracy': accuracy if ensemble_predictions else 0
    }

# -------------------------------
# 8. 可視化模型比較結果
# -------------------------------
def visualize_model_comparison(comparison_df, output_dir):
    """可視化不同模型的性能比較"""
    logger.info("生成模型比較可視化圖表")
    
    # 計算每個模型的正確率
    model_accuracies = {}
    for model in ['paddle', 'trocr', 'crnn']:
        correct_col = f'{model}_correct'
        if correct_col in comparison_df.columns:
            accuracy = comparison_df[correct_col].mean()
            model_accuracies[model] = accuracy * 100
    
    # 生成條形圖比較模型準確率
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    plt.bar(model_accuracies.keys(), model_accuracies.values(), color=colors)
    plt.title('不同OCR模型準確率比較', fontsize=15)
    plt.ylabel('準確率 (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # 添加數值標籤
    for i, (model, accuracy) in enumerate(model_accuracies.items()):
        plt.text(i, accuracy + 1, f'{accuracy:.2f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # 生成混淆矩陣熱圖，分析模型間的預測一致性
    if len(model_accuracies) >= 2:
        model_names = list(model_accuracies.keys())
        consistency_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    consistency_matrix[i, j] = 1.0
                else:
                    # 計算兩個模型預測一致的比例
                    pred1_col = f'{model1}_pred'
                    pred2_col = f'{model2}_pred'
                    
                    if pred1_col in comparison_df.columns and pred2_col in comparison_df.columns:
                        agreement = (comparison_df[pred1_col] == comparison_df[pred2_col]).mean()
                        consistency_matrix[i, j] = agreement
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(consistency_matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=model_names, yticklabels=model_names)
        plt.title('模型間預測一致性矩陣', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_consistency_matrix.png'), dpi=300)
        plt.close()
    
    # 如果有足夠的數據，繪製不同長度標籤的準確率比較
    if 'true_label' in comparison_df.columns:
        comparison_df['label_length'] = comparison_df['true_label'].str.len()
        
        length_accuracies = {}
        label_lengths = comparison_df['label_length'].unique()
        
        for model in model_accuracies.keys():
            correct_col = f'{model}_correct'
            if correct_col in comparison_df.columns:
                length_accuracies[model] = [
                    comparison_df[comparison_df['label_length'] == length][correct_col].mean() * 100
                    for length in label_lengths
                ]
        
        if length_accuracies:
            plt.figure(figsize=(12, 6))
            bar_width = 0.25
            index = np.arange(len(label_lengths))
            
            for i, (model, accuracies) in enumerate(length_accuracies.items()):
                plt.bar(index + i*bar_width, accuracies, bar_width, label=model)
            
            plt.xlabel('標籤長度')
            plt.ylabel('準確率 (%)')
            plt.title('不同長度標籤的準確率比較')
            plt.xticks(index + bar_width, label_lengths)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_by_label_length.png'), dpi=300)
            plt.close()

# -------------------------------
# 9. 預測函數
# -------------------------------
def predict_with_models(data_samples, model_dir, output_dir, args):
    """使用訓練好的模型進行預測"""
    logger.info("=== 使用訓練好的模型進行預測 ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到所有模型fold
    fold_dirs = [d for d in os.listdir(model_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(model_dir, d))]
    if not fold_dirs:
        logger.error(f"在 {model_dir} 中沒有找到模型文件夾")
        return
    
    # 初始化各模型的預測結果
    all_predictions = {
        'paddle': {},
        'trocr': {},
        'crnn': {}
    }
    
    # 使用每個fold的模型進行預測
    for fold_dir in fold_dirs:
        fold_path = os.path.join(model_dir, fold_dir)
        logger.info(f"使用 {fold_dir} 的模型進行預測")
        
        # 為每個模型類型進行預測
        for model_type in ['paddle', 'trocr', 'crnn']:
            model_path = os.path.join(fold_path, model_type)
            if not os.path.exists(model_path):
                logger.warning(f"找不到 {model_type} 模型路徑: {model_path}")
                continue
                
            logger.info(f"載入 {model_type} 模型")
            
            try:
                # 根據模型類型設置轉換和數據集
                if model_type == 'paddle':
                    transform = None
                    model = PaddleOCRWrapper(use_gpu=torch.cuda.is_available())
                elif model_type == 'trocr':
                    transform = transforms.Compose([
                        transforms.Resize((384, 384)),
                        transforms.ToTensor()
                    ])
                    model_path = os.path.join(model_path, "best_trocr_model")
                    if os.path.exists(model_path):
                        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                        model = TrOCRWrapper(model_name=model_path)
                    else:
                        logger.warning(f"找不到TrOCR模型: {model_path}")
                        continue
                elif model_type == 'crnn':
                    transform = transforms.Compose([
                        transforms.Resize((32, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    model_path = os.path.join(model_path, "best_crnn_model")
                    if os.path.exists(model_path):
                        model = CRNNWrapper(model_name=model_path)
                    else:
                        logger.warning(f"找不到CRNN模型: {model_path}")
                        continue
                
                # 創建數據集
                test_dataset = OCRDataset(data_samples, transform=transform, model_type=model_type)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                # 進行預測
                for images, _, frames in tqdm(test_loader, desc=f"預測 {model_type}"):
                    for i, (image, frame) in enumerate(zip(images, frames)):
                        pred = model.predict(image)
                        frame_num = int(frame.item()) if hasattr(frame, 'item') else frame
                        
                        if frame_num not in all_predictions[model_type]:
                            all_predictions[model_type][frame_num] = []
                        
                        all_predictions[model_type][frame_num].append(pred)
                
            except Exception as e:
                logger.error(f"使用 {model_type} 模型預測時出錯: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # 對每個模型，合併所有fold的預測結果（取眾數或加權平均）
    final_predictions = {}
    
    for model_type in all_predictions:
        model_preds = {}
        
        for frame, preds in all_predictions[model_type].items():
            if preds:
                # 取眾數作為最終預測
                pred_counts = defaultdict(int)
                for pred in preds:
                    pred_counts[pred.strip()] += 1
                
                most_common = max(pred_counts.items(), key=lambda x: x[1])[0]
                model_preds[frame] = most_common
        
        final_predictions[model_type] = model_preds
    
    # 創建集成預測
    ensemble_predictions = {}
    
    # 載入全局模型準確率作為權重（如果有）
    weights = {}
    ensemble_info_path = os.path.join(model_dir, "all_models_results.json")
    
    if os.path.exists(ensemble_info_path):
        try:
            with open(ensemble_info_path, 'r') as f:
                all_results = json.load(f)
                
            for model_type, results in all_results.items():
                if results:
                    accuracies = [r['accuracy'] for r in results]
                    weights[model_type] = np.mean(accuracies)
        except Exception as e:
            logger.warning(f"載入模型權重時出錯: {e}")
    
    # 如果沒有權重信息，使用均等權重
    if not weights:
        weights = {model_type: 1.0 for model_type in final_predictions}
    
    # 輸出模型權重
    logger.info("模型權重:")
    for model_type, weight in weights.items():
        logger.info(f"  - {model_type}: {weight:.4f}")
    
    # 對每個frame進行集成預測
    all_frames = set()
    for model_preds in final_predictions.values():
        all_frames.update(model_preds.keys())
    
    for frame in sorted(all_frames):
        # 收集各模型的預測
        frame_preds = {}
        for model_type, model_preds in final_predictions.items():
            if frame in model_preds:
                frame_preds[model_type] = model_preds[frame]
        
        if not frame_preds:
            continue
        
        # 加權投票
        pred_votes = defaultdict(float)
        for model, pred in frame_preds.items():
            pred_votes[pred] += weights.get(model, 0)
        
        # 選擇得票最高的預測
        if pred_votes:
            ensemble_pred = max(pred_votes.items(), key=lambda x: x[1])[0]
            ensemble_predictions[frame] = ensemble_pred
    
    # 保存每個模型的預測結果
    for model_type, model_preds in final_predictions.items():
        if model_preds:
            pred_list = [{'frame': frame, 'prediction': pred} for frame, pred in sorted(model_preds.items())]
            
            with open(os.path.join(output_dir, f"{model_type}_predictions.json"), 'w', encoding='utf-8') as f:
                json.dump({'predictions': pred_list}, f, ensure_ascii=False, indent=2)
            
            # 創建CSV
            pred_df = pd.DataFrame(pred_list)
            pred_df.to_csv(os.path.join(output_dir, f"{model_type}_predictions.csv"), index=False)
    
    # 保存集成預測結果
    if ensemble_predictions:
        ensemble_list = [{'frame': frame, 'prediction': pred} for frame, pred in sorted(ensemble_predictions.items())]
        
        with open(os.path.join(output_dir, "ensemble_predictions.json"), 'w', encoding='utf-8') as f:
            json.dump({'predictions': ensemble_list}, f, ensure_ascii=False, indent=2)
        
        # 創建CSV
        ensemble_df = pd.DataFrame(ensemble_list)
        ensemble_df.to_csv(os.path.join(output_dir, "ensemble_predictions.csv"), index=False)
    
    logger.info(f"預測完成，結果已保存到 {output_dir}")
    return ensemble_predictions

# -------------------------------
# 10. 主函數
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="使用多種預訓練OCR模型進行微調與預測")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train", "predict", "validate_all"],
                        help="操作模式: train (訓練), predict (預測), validate_all (整合驗證結果)")
    parser.add_argument("--jsonl", type=str, nargs='+', default=["region2/region2.jsonl"],
                        help="JSONL文件路徑，可指定多個文件")
    parser.add_argument("--base-dir", type=str, default="",
                        help="圖像路徑的基礎目錄")
    parser.add_argument("--output-dir", type=str, default="ocr_models",
                        help="輸出目錄")
    parser.add_argument("--model-dir", type=str, default="ocr_models",
                        help="模型目錄（predict和validate_all模式需要）")
    parser.add_argument("--epochs", type=int, default=10,
                        help="訓練輪數（對於可訓練的模型）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批量大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子")
    args = parser.parse_args()
    
    # 設置隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 載入數據
    data_samples = load_data_from_jsonl(args.jsonl, args.base_dir)
    if not data_samples:
        logger.error("沒有找到有效的數據樣本")
        return
    
    # 根據模式執行相應操作
    if args.mode == "train":
        logger.info("=== 開始訓練多模型OCR系統 ===")
        train_with_cross_validation(data_samples, args.output_dir, args)
        
    elif args.mode == "predict":
        logger.info("=== 開始使用多模型OCR系統預測 ===")
        predict_with_models(data_samples, args.model_dir, args.output_dir, args)
        
    elif args.mode == "validate_all":
        logger.info("=== 開始整合所有驗證集結果 ===")
        collect_all_validation_predictions(data_samples, args.model_dir, args.output_dir, args)
        
    logger.info("處理完成！")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"總執行時間: {(end_time - start_time)/60:.2f} 分鐘")