import sys
import os 
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import easyocr

# 創建 EasyOCR 閱讀器，設置 allowlist_chars 只包含數字、空格和負號
reader = easyocr.Reader(['en'], gpu=True)

# 讀取並識別圖像
result = reader.readtext(sys.argv[1], allowlist='0123456789- ')

# 後處理結果，確保輸出格式正確
processed_results = []
for bbox, text, prob in result:
    # 只保留置信度大於0.7的結果
    # if prob < 0.5:
    #     continue
        
    # 清理文本，只保留數字、空格和負號
    cleaned_text = ''.join(c for c in text if c.isdigit() or c == '-' or c == ' ')
    
    # 清理多餘空格
    cleaned_text = ' '.join(cleaned_text.split())
    
    # 確保數字格式正確 (如負號後必須跟數字)
    parts = cleaned_text.split()
    valid_parts = []
    for part in parts:
        # 處理獨立的負號
        if part == '-':
            continue
            
        # 處理數字與負號
        if part.startswith('-') and len(part) > 1:
            if part[1].isdigit():  # 確保負號後是數字
                valid_parts.append(part)
        elif part.isdigit():  # 純數字
            valid_parts.append(part)
    
    # 合併有效部分
    final_text = ' '.join(valid_parts)
    
    # 只有非空文本才添加到結果中
    if final_text:
        processed_results.append((bbox, final_text, prob))

# 按照左上角x坐標排序結果（從左到右）
processed_results.sort(key=lambda x: x[0][0][0])  # x[0]是bbox, x[0][0]是左上角坐標, x[0][0][0]是左上角x坐標

# 輸出處理後的結果
print(processed_results)

# 如果只需要文本部分，可以這樣輸出：
if processed_results:
    text_only = ' '.join([text for _, text, _ in processed_results])
    print("識別的數字: ", text_only)
else:
    print("未識別到有效數字")
