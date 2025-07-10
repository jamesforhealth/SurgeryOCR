"""
OCR模型接口包 - 簡潔統一的OCR接口
"""
from .base import BaseOCRModel
from .easyocr_model import EasyOCRModel
import torch
import os

# 檢查CRNN模型所需的依賴是否可用
try:
    from .simpleocr_model import CRNNModel
    has_crnn = True
except ImportError as e:
    print(f"DEBUG: 在 __init__.py 中導入 '.simpleocr_model' 時捕獲到 ImportError: {e}")
    print(f"DEBUG: 這通常意味著 simpleocr_model.py 或其依賴 (如 pretrain_crnn.py) 導入失敗。")
    has_crnn = False
except Exception as ex:
    print(f"DEBUG: 在 __init__.py 中導入 '.simpleocr_model' 時捕獲到非 ImportError 的異常: {ex}")
    has_crnn = False

# 預訓練模型路徑配置
PRETRAINED_MODELS = {
    "crnn_default": "models/OCR_interface/simpleocr/best_crnn_model.pth",
    "crnn_fine_tuned": "models/OCR_interface/simpleocr/best_finetuned_model.pth",
}

def get_ocr_model(model_type="easyocr", pretrained=None, debug_output=True, **kwargs):
    """
    獲取OCR模型實例
    
    參數:
        model_type: 模型類型，支持 'easyocr' 或 'crnn'
        pretrained: 使用預訓練模型名稱，例如 'default' 或 'fine_tuned'，
                   如果指定，將忽略model_path參數
        debug_output: 是否在終端輸出詳細的調試信息
        **kwargs: 傳遞給模型初始化函數的參數
    
    返回:
        OCR模型實例
    """
    if model_type.lower() == "easyocr":
        # 將debug_output參數傳遞給EasyOCRModel
        kwargs['debug_output'] = debug_output
        return EasyOCRModel(**kwargs)
    
    elif model_type.lower() == "crnn":
        if not has_crnn:
            raise ImportError("使用CRNN模型需要安裝PyTorch (或者其依賴導入失敗，請查看上面的 DEBUG 信息)")
        
        # 檢查是否使用預訓練模型
        if pretrained:
            model_key = f"crnn_{pretrained}"
            if model_key in PRETRAINED_MODELS:
                kwargs['model_path'] = PRETRAINED_MODELS[model_key]
            else:
                raise ValueError(f"找不到預訓練模型: {pretrained}，可用選項: {list(PRETRAINED_MODELS.keys())}")
        
        crnn_model_path = kwargs.get("model_path")
        crnn_characters = kwargs.get("characters") # 可能為 None
        pretrained_flag = kwargs.get("pretrained")

        DEFAULT_PRETRAINED_MODEL_PATH = "models/weights/crnn_general_pretrained.pth" # 您需要提供這個文件
        DEFAULT_PRETRAINED_CHARACTERS = "0123456789abcdefghijklmnopqrstuvwxyz" # 與上述權重匹配的字符集

        if pretrained_flag == "default":
            if crnn_model_path:
                print(f"警告：同時指定了 pretrained='default' 和 model_path='{crnn_model_path}'。將優先使用 model_path。")
            else:
                if not os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    raise FileNotFoundError(f"CRNN 'default' 預訓練權重 {DEFAULT_PRETRAINED_MODEL_PATH} 未找到!")
                crnn_model_path = DEFAULT_PRETRAINED_MODEL_PATH
                if crnn_characters is None: # 只有當用戶沒有通過 --crnn_characters 指定時，才使用默認權重的字符集
                    crnn_characters = DEFAULT_PRETRAINED_CHARACTERS
                print(f"使用 CRNN 默認預訓練模型: {crnn_model_path}")

        if not crnn_model_path:
            raise ValueError("CRNN模型需要 'model_path' 參數，或者指定 pretrained='default' 且默認權重可用。")

        return CRNNModel(
            model_path=crnn_model_path,
            characters=crnn_characters, # CRNNModel 內部會處理 None
            gpu=kwargs.get('device', 'cpu') == 'cuda'
        )
    
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")

# 提供簡單方便的函數來直接識別圖像
def recognize_text(image, model_type="easyocr", pretrained=None, **kwargs):
    """
    簡便函數：直接識別圖像中的文字
    
    參數:
        image: 輸入圖像，numpy數組 (BGR)
        model_type: 模型類型，支持 'easyocr' 或 'crnn'
        pretrained: 使用預訓練模型，支持 'default' 或 'fine_tuned'
        **kwargs: 其他參數
    
    返回:
        text: 識別的文字
        confidence: 置信度
    """
    model = get_ocr_model(model_type=model_type, pretrained=pretrained, **kwargs)
    return model.recognize(image)
