import argparse
import json
import os
import re
import string
import requests
import torch

from PIL import Image
from io import BytesIO
import cv2
import numpy as np  # For image concatenation

# Transformers and GOT-related imports
from transformers import AutoTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor
from GOT.model import GOTQwenForCausalLM
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from GOT.demo.process_results import punctuation_dict, svg_to_html

# Constants used in the script
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

# Translation table for punctuation
translation_table = str.maketrans(punctuation_dict)

def load_regions(json_path):
    """
    Load region information from a JSON file.
    Returns a list of region configurations, each containing:
    {
        "coords": [x1, y1, x2, y2],
        "binarize": bool,
        "threshold": int,
        "allowed_chars": str or None,
        "invert": bool,
        "default_content": str
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        regions = json.load(f)
    return regions

def load_region8_jsonl(jsonl_path):
    """
    Load all lines from region8.jsonl into a list of dicts.
    Each line is one JSON object with fields like:
        {"query": "...", "response": "...", "images": "..."}
    Returns a list of dicts.
    """
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(data)
            except json.JSONDecodeError:
                # If a line is malformed, skip or handle differently
                continue
    return results

def load_digit_counts_from_jsonl(jsonl_path):
    """
    從JSONL文件中讀取每個幀的數字個數信息。
    返回一個字典，鍵為幀號，值為該幀中的數字個數。
    """
    digit_counts = {}
    
    if not os.path.exists(jsonl_path):
        print(f"警告: 找不到標籤文件 {jsonl_path}")
        return digit_counts
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 從圖像路徑中提取幀號
                image_path = data.get('images', '')
                if 'frame_' in image_path:
                    frame_num = int(re.search(r'frame_(\d+)', image_path).group(1))
                    # 從response中計算數字個數
                    response = data.get('response', '')
                    digit_count = len(re.findall(r'\d', response))
                    digit_counts[frame_num] = digit_count
            except (json.JSONDecodeError, AttributeError) as e:
                continue
    
    print(f"從 {jsonl_path} 讀取了 {len(digit_counts)} 個幀的數字個數信息")
    return digit_counts

def process_region(frame_region, binarize=False, threshold=150, invert=False):
    """
    Given a cropped frame region (as a NumPy array in BGR or grayscale),
    optionally perform binarization, inversion, and then
    convert the result to a PIL Image for OCR.
    """
    if binarize:
        if len(frame_region.shape) == 3 and frame_region.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_region

        _, frame_gray = cv2.threshold(frame_gray, threshold, 255, cv2.THRESH_BINARY)
        if invert:
            frame_gray = cv2.bitwise_not(frame_gray)

        frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
    else:
        if len(frame_region.shape) == 3 and frame_region.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_region, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_region, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(frame_rgb)
    return pil_img

def process_image(args, image, model, tokenizer,
                  image_processor, image_processor_high,
                  use_im_start_end=True, image_token_len=256, digit_count=None):
    """
    Given a PIL Image, perform inference and return the OCR result (string).
    
    Parameters:
        args: Command line arguments
        image: PIL Image to process
        model: The OCR model
        tokenizer: Tokenizer for the model
        image_processor: Image processor for low-res
        image_processor_high: Image processor for high-res
        use_im_start_end: Whether to use image start/end tokens
        image_token_len: Length of image token sequence
        digit_count: Optional. If provided, specifies the number of digits expected in the image
    """
    # 依據 args.type 和 digit_count 調整提示詞
    if args.type == 'format':
        qs = 'OCR with format: '
    else:
        if digit_count is not None:
            if digit_count == 0:
                #圖像中沒有數字。不要輸出任何數字 process_image直接回傳結果
                return ""
            elif digit_count == 1:
                qs = '請識別以下圖像中的那個數字： OCR: '
            elif digit_count == 2:
                qs = '請以左邊到右邊的順序識別以下圖像中的兩個數字： OCR: '
            elif digit_count == 3:
                qs = '請以左邊到右邊的順序識別以下圖像中的三個數字： OCR: '
            # else:
            #     qs = f'圖像中有{digit_count}個數字。請從左到右識別並依序輸出這些數字: '
        else:
            qs = '請按照從左到右的順序識別圖像中的數字： OCR: ' 

    if use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            + DEFAULT_IM_END_TOKEN
            + '\n'
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"
    # 如果之後還需要用到 conv_mode，可保留；否則可以刪除
    args.conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    image_tensor = image_processor(image)
    image_tensor_1 = image_processor_high(image)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(),
                     image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs

def handle_key_press(cap, frame_count, paused):
    """
    Listen for key presses:
      'q' = quit
      ' ' = pause/resume
      'a' = rewind 10 frames
      'd' = fast-forward 10 frames
    """
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        return True, frame_count, paused
    elif key == ord(' '):
        paused = not paused
    elif key == ord('a'):
        new_frame = max(0, frame_count - 10)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        ret, jumped_frame = cap.read()
        if ret:
            frame_count = new_frame
            resized_frame = cv2.resize(jumped_frame, (640, 360))
            cv2.imshow("video", resized_frame)
    elif key == ord('d'):
        new_frame = frame_count + 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        ret, jumped_frame = cap.read()
        if ret:
            frame_count = new_frame
            resized_frame = cv2.resize(jumped_frame, (640, 360))
            cv2.imshow("video", resized_frame)

    return False, frame_count, paused

def eval_model(args):
    # Disable default torch initialization
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map='cuda',
        use_safetensors=True,
        pad_token_id=151643
    ).eval()
    model.to(device='cuda', dtype=torch.bfloat16)

    # Image pre-processing
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    # Load regions from JSON
    if not hasattr(args, 'region_file') or not args.region_file:
        print("No region_file specified. Using the entire frame for OCR.")
        regions = []
    else:
        regions = load_regions(args.region_file)
        print(f"Loaded {len(regions)} region(s) from {args.region_file}.")

    # groupA = relevant regions for output folders + certain OCR stages
    #groupA = [1, 2, 3, 6, 7]  # "relevant_region_indices"
    groupA = [2]
    # groupB = subset used for certain other OCR stages
    #groupB = [1, 2, 3]
    groupB = [2]

    # 載入數字個數標籤（如果提供了標籤文件）
    digit_counts = {}
    if args.label_file:
        digit_counts = load_digit_counts_from_jsonl(args.label_file)

    # Create folders and JSONL files only for groupA
    region_jsonl_files = [None] * len(regions)
    for i in range(len(regions)):
        if i in groupA:
            region_dir = f"region{i}"
            os.makedirs(region_dir, exist_ok=True)
            jsonl_path = os.path.join(region_dir, f"region{i}_test_adaptive.jsonl") #os.path.join(region_dir, f"region{i}.jsonl")
            region_jsonl_files[i] = open(jsonl_path, 'a', encoding='utf-8')

    # Load region8.jsonl for stage logic (but do NOT create a folder for region8)
    region8_entries = []
    if args.region8_file and os.path.exists(args.region8_file):
        region8_entries = load_region8_jsonl(args.region8_file)
        print(f"Loaded {len(region8_entries)} record(s) from {args.region8_file}.")
    else:
        print("No region8_file specified or file does not exist.")
    
    # Track stage
    stage = ""

    print(f"Reading video {args.video_file}, performing OCR with stage-based logic...")
    cap = cv2.VideoCapture(args.video_file)
    frame_count = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # 1) Update 'stage' from region8.jsonl, if applicable
            if frame_count < len(region8_entries):
                response8 = region8_entries[frame_count].get("response", "")
                if response8 != "":
                    stage = response8
                else:
                    if stage == "" and response8 == "":
                        stage = ""

            # If no regions, do OCR on entire frame
            if len(regions) == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                result = process_image(
                    args,
                    pil_frame,
                    model,
                    tokenizer,
                    image_processor,
                    image_processor_high,
                    use_im_start_end=True,
                    image_token_len=256
                )
                print(f"[Frame {frame_count}] OCR Result: {result}")

                frame_count += 1
                resized_frame = cv2.resize(frame, (640, 360))
                cv2.imshow("video", resized_frame)
            else:
                # Prepare to store final "response" for each region
                region_responses = [""] * len(regions)

                def do_ocr_for_region(idx):
                    """
                    Return True if we should perform OCR on region idx
                    given the current stage; otherwise False.
                    """
                    # Stages that SKIP groupA:
                    if stage in ["Position", "Registration", "Capsulorhexis"]:
                        if idx in groupA:
                            return False

                    # Stages that DO OCR for groupA:
                    elif stage in ["Sculpt", "Quad"]:
                        if idx in groupA:
                            return True

                    # Stages that DO OCR for groupB, skip region6,7:
                    elif stage in ["Epi", "Cortex", "Polish", "Toric", "Visco", "Centration"]:
                        if idx in groupB:
                            return True
                        elif idx in [6, 7]:
                            return False

                    # Otherwise skip
                    return False

                # Loop over all regions
                for idx, reg in enumerate(regions):
                    if do_ocr_for_region(idx):
                        x1, y1, x2, y2 = reg["coords"]
                        sub_frame = frame[y1:y2, x1:x2]

                        region_image = process_region(
                            sub_frame,
                            binarize=reg["binarize"],
                            threshold=reg["threshold"],
                            invert=reg["invert"]
                        )
                        
                        # 獲取當前幀的數字個數（如果有）
                        current_digit_count = None
                        if frame_count in digit_counts:
                            current_digit_count = digit_counts[frame_count]
                            print(f"[Frame {frame_count}, Region {idx}] 使用標籤的數字個數: {current_digit_count}")
                        
                        # 如果數字個數為0，直接設置結果為空字符串，跳過OCR處理
                        if current_digit_count == 0:
                            result = ""
                            print(f"[Frame {frame_count}, Region {idx}] 數字個數為0，跳過OCR處理")
                        else:
                            result = process_image(
                                args,
                                region_image,
                                model,
                                tokenizer,
                                image_processor,
                                image_processor_high,
                                use_im_start_end=True,
                                image_token_len=256,
                                digit_count=current_digit_count
                            )

                            # 只有在非零數字個數的情況下才應用字符過濾
                            allowed_chars = reg["allowed_chars"]
                            if allowed_chars:
                                result = "".join(ch for ch in result if ch in allowed_chars)

                        # 只有在結果為空且數字個數不為0的情況下才使用default_content
                        if not result.strip() and (current_digit_count is None or current_digit_count > 0):
                            result = reg["default_content"]

                        region_responses[idx] = result
                        print(f"[Frame {frame_count}, Region {idx}, Stage={stage}] OCR Result: {result}")
                    else:
                        region_responses[idx] = ""

                # Write each region's updated response to JSONL if it's in groupA
                for idx, reg in enumerate(regions):
                    if idx in groupA and region_jsonl_files[idx] is not None:
                        # If region_responses[idx] != "", we performed OCR
                        if region_responses[idx] != "":
                            x1, y1, x2, y2 = reg["coords"]
                            sub_frame = frame[y1:y2, x1:x2]
                            region_image_cv = sub_frame  # BGR

                            region_dir = f"region{idx}"
                            region_image_filename = f"frame_{frame_count}.png"
                            region_image_path = os.path.join(region_dir, region_image_filename)
                            cv2.imwrite(region_image_path, region_image_cv)

                            # 添加數字個數信息到JSON記錄中
                            digit_count_info = None
                            if frame_count in digit_counts:
                                digit_count_info = digit_counts[frame_count]
                                
                            json_record = {
                                "query": "<image>",
                                "response": region_responses[idx],
                                "images": f"{region_dir}/{region_image_filename}",
                                "digit_count": digit_count_info
                            }
                            region_jsonl_files[idx].write(json.dumps(json_record, ensure_ascii=False) + "\n")

                # Draw bounding boxes (optional)
                for idx, reg in enumerate(regions):
                    x1, y1, x2, y2 = reg["coords"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                frame_count += 1
                resized_frame = cv2.resize(frame, (640, 360))
                cv2.imshow("video", resized_frame)

        should_break, frame_count, paused = handle_key_press(cap, frame_count, paused)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Close any open JSONL files
    for f in region_jsonl_files:
        if f is not None:
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m",
                        help="Name of the model to use for OCR")
    parser.add_argument("--video-file", type=str, required=True, help="Path to the video file")
    parser.add_argument("--region-file", type=str, default='',
                        help="Path to the regions.json file for region-based OCR")
    parser.add_argument("--region8-file", type=str, default='',
                        help="Path to the region8.jsonl file (contains 'response' for stage control)")
    parser.add_argument("--label-file", type=str, default='',
                        help="Path to the label JSONL file containing digit count information")
    # 只保留 --type
    parser.add_argument("--type", type=str, required=True,
                        help="OCR mode, can be: normal or format")
    args = parser.parse_args()

    eval_model(args)
