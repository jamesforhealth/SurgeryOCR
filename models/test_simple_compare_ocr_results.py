import json

# 載入兩種方法的結果
with open('results_classification/prediction_results.json', 'r') as f:
    class_results = json.load(f)

with open('results_sequence/prediction_results.json', 'r') as f:
    seq_results = json.load(f)

# 比較準確率
print(f"分類法準確率: {class_results['statistics']['accuracy']*100:.2f}%")
print(f"序列法準確率: {seq_results['statistics']['accuracy']*100:.2f}%")

# 找出兩種方法預測結果不同的案例
different_predictions = []

class_preds = {p['frame']: p for p in class_results['predictions']}
seq_preds = {p['frame']: p for p in seq_results['predictions']}

for frame in set(class_preds.keys()) & set(seq_preds.keys()):
    class_pred = class_preds[frame]['prediction']
    seq_pred = seq_preds[frame]['prediction']
    true_label = class_preds[frame]['true_label']
    
    if class_pred != seq_pred:
        different_predictions.append({
            'frame': frame,
            'true_label': true_label,
            'classification_prediction': class_pred,
            'sequence_prediction': seq_pred,
            'classification_correct': class_pred == true_label,
            'sequence_correct': seq_pred == true_label
        })

# 保存差異結果
with open('method_comparison.json', 'w') as f:
    json.dump({
        'different_predictions_count': len(different_predictions),
        'only_classification_correct': sum(1 for p in different_predictions if p['classification_correct'] and not p['sequence_correct']),
        'only_sequence_correct': sum(1 for p in different_predictions if not p['classification_correct'] and p['sequence_correct']),
        'both_incorrect': sum(1 for p in different_predictions if not p['classification_correct'] and not p['sequence_correct']),
        'details': sorted(different_predictions, key=lambda x: x['frame'])
    }, f, indent=2)

print(f"兩種方法的比較結果已保存到 method_comparison.json")