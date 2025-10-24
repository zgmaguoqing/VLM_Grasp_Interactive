import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor

import whisper
import json
import re
import base64
import textwrap
import queue
import time
import io

import soundfile as sf  
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment

from openai import OpenAI  # 导入OpenAI客户端

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ----------------------- 基础工具函数 -----------------------

def encode_np_array(image_np):
    """将 numpy 图像数组（BGR）编码为 base64 字符串"""
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("无法将图像数组编码为 JPEG")
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64



# ----------------------- 多模态模型调用（Qwen） -----------------------

def generate_robot_actions(user_command, image_input=None):
    """
    使用 base64 的方式将 numpy 图像和用户文本指令传给 Qwen 多模态模型，
    要求模型返回两部分：
      - 模型返回内容中，第一部分为自然语言响应（说明为何选择该物体），
      - 紧跟其后的部分为纯 JSON 对象，格式如下：

        {
          "name": "物体名称",
          "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
        }

    返回一个 dict，包含 "response" 和 "coordinates"。
    参数 image_input 为 numpy 数组（BGR 格式）。
    """
    # 初始化OpenAI客户端
    # 替换为自己的模型调用，没有本地部署的，可以参考该网站 https://sg.uiuiapi.com/v1
    client = OpenAI(api_key='sk-21a42456345b4b6da1df9d0a08a4396c', base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    system_prompt = textwrap.dedent("""\
    你是一个精密机械臂视觉控制系统，具备先进的多模态感知能力。请严格按照以下步骤执行任务：

    【图像分析阶段】
    1. 分析输入图像，识别图像中所有可见物体，并记录每个物体的边界框（左上角点和右下角点）及其类别名称。

    【指令解析阶段】
    2. 根据用户的自然语言指令，从识别的物体中筛选出最匹配的目标物体。

    【响应生成阶段】
    3. 输出格式必须严格如下：
    - 自然语言响应（仅包含说明为何选择该物体的文字,可以俏皮可爱地回应用户的需求，但是请注意，回答中应该只包含被选中的物体），
    - 紧跟其后，从下一行开始返回 **标准 JSON 对象**,但是不要返回json本体,格式如下：

    {
      "name": "物体名称",
      "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
    }

    【注意事项】
    - JSON 必须从下一行开始；
    - 自然语言响应与 JSON 之间无其他额外文本;
    - JSON 对象不能有任何注释、额外文本或解释,包括不能有辅助标识为json文本的内容,不要有json;
    - 坐标 bbox 必须为整数；
    - 只允许使用 "bbox" 作为坐标格式。
    """)

    messages = [{"role": "system", "content": system_prompt}]
    user_content = []

    if image_input is not None:
        base64_img = encode_np_array(image_input)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    user_content.append({"type": "text", "text": user_command})
    messages.append({"role": "user", "content": user_content})

    try:
        # 使用OpenAI客户端调用API
        completion = client.chat.completions.create(
            # model="qwen-vl-plus",  # 指定模型名称，请确认服务提供商支持的模型名
            model="qwen-vl-max", 
            # model="gpt-5",
            # model="qwen2.5-vl-32b-instruct",
            messages=messages,
            # max_tokens=4096,  # 可根据需要调整
            temperature=0.1,   # 降低温度以提高输出的确定性，对结构化输出有益
        )
        
        content = completion.choices[0].message.content
        print("原始响应：", content)

        # 使用正则表达式查找 JSON 部分
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                coord = json.loads(json_str)
            except Exception as e:
                print(f"[警告] JSON 解析失败：{e}")
                coord = {}
            natural_response = content[:match.start()].strip()
        else:
            natural_response = content.strip()
            coord = {}

        return {
            "response": natural_response,
            "coordinates": coord
        }

    except Exception as e:
        print(f"请求失败：{e}")
        return {"response": "处理失败", "coordinates": {}}
# -----------
# import base64
# import json
# import re
# import textwrap
# from typing import Dict, Any, Optional, List

# import numpy as np

# # ============== 依赖（HF 本地推理） ==============
# # pip install "transformers>=4.44.0" accelerate torch pillow
# import torch
# from PIL import Image
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# # （可选）仍保留 OpenAI-兼容路径；如果只用 HF，可删除本段依赖与函数
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None


# # ---------------- 公共工具函数 ----------------
# def encode_np_array(img_bgr: np.ndarray) -> str:
#     """
#     将 BGR numpy 数组编码为 JPEG -> base64 字符串（用于 OpenAI-兼容后端）。
#     HF 后端不需要 base64，直接用 PIL 即可。
#     """
#     import cv2
#     ok, buf = cv2.imencode(".jpg", img_bgr)
#     if not ok:
#         raise ValueError("图像编码失败")
#     return base64.b64encode(buf.tobytes()).decode("utf-8")


# def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
#     """
#     将 BGR numpy 数组转换为 PIL(RGB)。
#     """
#     if img_bgr is None:
#         return None
#     img_rgb = img_bgr[..., ::-1].copy()
#     return Image.fromarray(img_rgb)


# def _build_system_prompt() -> str:
#     return textwrap.dedent("""\
#     你是一个精密机械臂视觉控制系统，具备先进的多模态感知能力。请严格按照以下步骤执行任务：

#     【图像分析阶段】
#     1. 分析输入图像，识别图像中所有可见物体，并记录每个物体的边界框（左上角点和右下角点）及其类别名称。

#     【指令解析阶段】
#     2. 根据用户的自然语言指令，从识别的物体中筛选出最匹配的目标物体。

#     【响应生成阶段】
#     3. 输出格式必须严格如下：
#     - 自然语言响应（仅包含说明为何选择该物体的文字,可以俏皮可爱地回应用户的需求，但是请注意，回答中应该只包含被选中的物体），
#     - 紧跟其后，从下一行开始返回 **标准 JSON 对象**,但是不要返回json本体,格式如下：

#     {
#       "name": "物体名称",
#       "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
#     }

#     【注意事项】
#     - JSON 必须从下一行开始；
#     - 自然语言响应与 JSON 之间无其他额外文本;
#     - JSON 对象不能有任何注释、额外文本或解释,包括不能有辅助标识为json文本的内容,不要有json;
#     - 坐标 bbox 必须为整数；
#     - 只允许使用 "bbox" 作为坐标格式。
#     """)


# def _extract_text_and_json(content: str) -> Dict[str, Any]:
#     """
#     从模型输出中分离自然语言部分与 JSON 对象。
#     """
#     match = re.search(r'(\{.*\})', content, re.DOTALL)
#     if match:
#         json_str = match.group(1)
#         try:
#             coord = json.loads(json_str)
#         except Exception as e:
#             print(f"[警告] JSON 解析失败：{e}")
#             coord = {}
#         natural_response = content[:match.start()].strip()
#     else:
#         natural_response = content.strip()
#         coord = {}

#     return {"response": natural_response, "coordinates": coord}


# # ---------------- OpenAI-兼容后端（可选保留） ----------------
# def _call_openai_compatible(
#     user_command: str,
#     image_b64: Optional[str],
#     model: str,
#     api_key: str = "",
#     base_url: str = ""
# ) -> Dict[str, Any]:
#     if OpenAI is None:
#         raise RuntimeError("未安装 openai 包，请 `pip install openai`。")
#     client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url or None)

#     system_prompt = _build_system_prompt()
#     messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

#     user_content: List[Dict[str, Any]] = []
#     if image_b64 is not None:
#         user_content.append({
#             "type": "image_url",
#             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
#         })
#     user_content.append({"type": "text", "text": user_command})
#     messages.append({"role": "user", "content": user_content})

#     completion = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.1
#     )
#     content = completion.choices[0].message.content
#     print("原始响应：", content)
#     return _extract_text_and_json(content)


# # ---------------- HF(Qwen3-VL) 本地后端 ----------------
# _HF_STATE: Dict[str, Any] = {"model": None, "processor": None, "model_name": None}


# def _ensure_hf_qwen3(
#     hf_model_name: str,
#     dtype: str = "auto",
#     device_map: str = "auto",
#     attn_implementation: Optional[str] = None
# ):
#     """
#     延迟加载/缓存 Qwen3-VL 模型与处理器。
#     """
#     if _HF_STATE["model"] is not None and _HF_STATE["model_name"] == hf_model_name:
#         return

#     kwargs = dict(dtype=dtype, device_map=device_map)
#     if attn_implementation:
#         kwargs["attn_implementation"] = attn_implementation

#     model = Qwen3VLForConditionalGeneration.from_pretrained(
#         hf_model_name, **kwargs
#     )
#     processor = AutoProcessor.from_pretrained(hf_model_name)

#     _HF_STATE["model"] = model
#     _HF_STATE["processor"] = processor
#     _HF_STATE["model_name"] = hf_model_name


# def _call_hf_qwen3(
#     user_command: str,
#     image_bgr: Optional[np.ndarray],
#     hf_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
#     max_new_tokens: int = 256,
#     attn_implementation: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """
#     使用本地 Transformers 直接调用 Qwen3-VL。
#     - 图像：传入 numpy BGR；内部转 PIL(RGB)
#     - 文本：遵循与原函数一致的 system prompt 与指令
#     """
#     _ensure_hf_qwen3(
#         hf_model_name=hf_model_name,
#         dtype="auto",
#         device_map="auto",
#         attn_implementation=attn_implementation,
#     )
#     model = _HF_STATE["model"]
#     processor = _HF_STATE["processor"]

#     system_prompt = _build_system_prompt()

#     # 组装 messages（Qwen3-VL 的聊天模板）
#     content_list: List[Dict[str, Any]] = []
#     if image_bgr is not None:
#         pil_img = bgr_to_pil(image_bgr)
#         content_list.append({"type": "image", "image": pil_img})
#     content_list.append({"type": "text", "text": user_command})

#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
#         {"role": "user",   "content": content_list},
#     ]

#     # 编码
#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         return_dict=True
#     )
#     # 将视觉张量等移动到相同设备
#     inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

#     # 生成（为保证结构化稳定，关闭采样）
#     with torch.no_grad():
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             use_cache=True
#         )

#     # 只解码新增的 token（去掉提示部分）
#     trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )[0]

#     print("原始响应：", output_text)
#     return _extract_text_and_json(output_text)


# # ---------------- 统一入口：保持原函数签名 ----------------
# def generate_robot_actions(
#     user_command: str,
#     image_input: Optional[np.ndarray] = None,
#     *,
#     # 选择后端："hf"（本地 Transformers）| "openai"（OpenAI-兼容）
#     backend: str = "openai",

#     # ---- HF(Qwen3-VL) 配置 ----
#     hf_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
#     hf_attn_implementation: Optional[str] = None,  # 如 "flash_attention_2"
#     hf_max_new_tokens: int = 256,

#     # ---- OpenAI-兼容配置（可选）----
#     api_key: str = "sk-21a42456345b4b6da1df9d0a08a4396c",
#     base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
#     model: str = "qwen-vl-plus",
# ) -> Dict[str, Any]:
#     """
#     使用 base64（OpenAI-兼容）或本地 Transformers(HF) 的方式与多模态模型对话，
#     并返回 {"response": 自然语言说明, "coordinates": JSON 坐标对象}。
#     """
#     try:
#         if backend == "hf":
#             return _call_hf_qwen3(
#                 user_command=user_command,
#                 image_bgr=image_input,
#                 hf_model_name=hf_model_name,
#                 max_new_tokens=hf_max_new_tokens,
#                 attn_implementation=hf_attn_implementation,
#             )
#         elif backend == "openai":
#             img_b64 = encode_np_array(image_input) if image_input is not None else None
#             return _call_openai_compatible(
#                 user_command=user_command,
#                 image_b64=img_b64,
#                 model=model,
#                 api_key=api_key,
#                 base_url=base_url
#             )
#         else:
#             raise ValueError("未知后端：应为 'hf' 或 'openai'。")

#     except Exception as e:
#         print(f"请求失败：{e}")
#         return {"response": "处理失败", "coordinates": {}}

# ----------------------- SAM 分割相关 -----------------------
def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)

def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# ----------------------- 语音识别与 TTS -----------------------

# 初始化全局模型变量
_global_models = {}


def load_models():
    """在需要时加载模型，避免启动时全部加载占用资源"""
    if not _global_models:
        print("🔄 正在加载离线语音模型...")
        # 加载Whisper小型模型 (适合你的6GB显存)
        # _global_models['asr'] = whisper.load_model("small")
        # _global_models['asr'] = whisper.load_model("tiny")
        # _global_models['asr'] = whisper.load_model("base")
        print("✅ Whisper的base模型加载完毕")

        try:
            import pyttsx3
            _global_models['tts_backup'] = pyttsx3.init()
            # 配置TTS
            _global_models['tts_backup'].setProperty('rate', 160)  # 语速
            voices = _global_models['tts_backup'].getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    _global_models['tts_backup'].setProperty('voice', voice.id)
                    break
            print("✅ TTS (pyttsx3) 初始化完毕")
        except Exception as e:
            print(f"⚠️  TTS初始化失败: {e}")
            _global_models['tts_backup'] = None

    return _global_models


# 音频参数配置
samplerate = 16000
channels = 1
dtype = 'int16'
frame_duration = 0.2
frame_samples = int(frame_duration * samplerate)
silence_threshold = 250
silence_max_duration = 2.0
q = queue.Queue()


def rms(audio_frame):
    samples = np.frombuffer(audio_frame, dtype=np.int16)
    if samples.size == 0:
        return 0
    mean_square = np.mean(samples.astype(np.float32) ** 2)
    if np.isnan(mean_square) or mean_square < 1e-5:
        return 0
    return np.sqrt(mean_square)

def callback(indata, frames, time_info, status):
    if status:
        print("⚠️ 状态警告：", status)
    q.put(bytes(indata))

def recognize_speech():
    """录音并返回音频数据（numpy 数组）"""
    print("🎙️ 启动录音，请说话...")
    # print("💡 调试信息：正在监测实时音量（RMS），请观察不说话时的基础噪音值")
    audio_buffer = []
    is_speaking = False
    last_voice_time = time.time()

    with sd.RawInputStream(samplerate=samplerate, blocksize=frame_samples,
                           dtype=dtype, channels=channels, callback=callback):
        while True:
            frame = q.get()
            volume = rms(frame)
            current_time = time.time()

            # print(f"实时音量（RMS）: {volume}") 

            if volume > silence_threshold:
                if not is_speaking:
                    print("🎤 检测到语音，开始录音...")
                    is_speaking = True
                    audio_buffer = []
                audio_np = np.frombuffer(frame, dtype=np.int16)
                audio_buffer.append(audio_np)
                last_voice_time = current_time
            elif is_speaking and (current_time - last_voice_time > silence_max_duration):
                print("🛑 停止录音，准备识别...")
                full_audio = np.concatenate(audio_buffer, axis=0)
                return full_audio
            elif not is_speaking and (current_time - last_voice_time > 10.0):
                print("🛑 超时：未检测到语音输入")
                return np.array([], dtype=np.int16)

def speech_to_text_offline(audio_data):
    """
    使用离线Whisper模型将录音数据转换为文本
    """
    print("📡 正在进行离线语音识别...")
    models = load_models()
    asr_model = models['asr']

    # 保存临时音频文件
    temp_wav = "temp_audio.wav"
    write(temp_wav, samplerate, audio_data.astype(np.int16))

    try:
        # 使用Whisper进行识别，指定语言为中文以提高精度和速度
        result = asr_model.transcribe(temp_wav, language="zh", fp16=torch.cuda.is_available())
        return result["text"].strip()
    except Exception as e:
        print(f"❌ 离线语音识别失败: {e}")
        return ""

def play_tts_offline(text):
    """
    使用离线TTS模型将文本转换为语音并播放
    """
    if not text:
        return
        
    print(f"📢 离线TTS播放: {text}")
    models = load_models()

    try:
        if models['tts_backup'] is not None:
            models['tts_backup'].say(text)
            models['tts_backup'].runAndWait()

    except Exception as e:
        print("❌ 无可用TTS引擎")


def voice_command_to_keyword():
    """
    获取语音命令并转换为文本。
    直接返回识别的文本指令。
    """
    audio_data = recognize_speech()
    text = speech_to_text_offline(audio_data) # 改为调用离线ASR
    if not text:
        print("⚠️ 没有识别到文本")
        return ""
    print("📝 识别文本：", text)
    # play_tts_offline(f"已收到指令: {text}") # 改为调用离线TTS
    return text


# ----------------------- 主流程：图像分割 -----------------------
def segment_image(image_input, output_mask='mask1.png'):
    # 1. 使用文字获取目标指令
    print("📝 请通过文字描述目标物体及抓取指令...")
    command_text = input("请输入: ").strip()
    if not command_text:
        print("⚠️ 未识别到语音指令，请重试。")
        return None
    print(f"✅ 识别的语音指令：{command_text}")

    # # 1. 使用语音获取目标指令
    # print("🎙️ 请通过语音描述目标物体及抓取指令...")
    # command_text = voice_command_to_keyword()
    # if not command_text:
    #     print("⚠️ 未识别到语音指令，请重试。")
    #     return None
    # print(f"✅ 识别的语音指令：{command_text}")

    # 2. 通过多模态模型获取检测框
    result = generate_robot_actions(command_text, image_input)
    natural_response = result["response"]
    detection_info = result["coordinates"]
    print("自然语言回应：", natural_response)
    print("检测到的物体信息：", detection_info)

    # 仅对模型返回的自然语言回应播报
    play_tts_offline(natural_response)
    
    bbox = detection_info.get("bbox") if detection_info and "bbox" in detection_info else None
    
    # 3. 准备图像供 SAM 使用（转换为 RGB）
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    # 4. 初始化 SAM，并设置图像
    predictor = choose_model()
    predictor.set_image(image_rgb)

    if bbox:
        results = predictor(bboxes=[bbox])
        center, mask = process_sam_results(results)
        print(f"✅ 自动检测到目标,bbox:{bbox}")
    else:
        print("⚠️ 未检测到目标，请点击图像选择对象")
        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', image_input)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                print(f"🖱️ 点击坐标：{x}, {y}")
                cv2.setMouseCallback('Select Object', lambda *args: None)

        cv2.setMouseCallback('Select Object', click_handler)
        while True:
            key = cv2.waitKey(100)
            if point:
                break
            if cv2.getWindowProperty('Select Object', cv2.WND_PROP_VISIBLE) < 1:
                print("❌ 窗口被关闭，未进行点击")
                return None
        cv2.destroyAllWindows()
        results = predictor(points=[point], labels=[1])
        center, mask = process_sam_results(results)

    # 5. 保存分割掩码
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"✅ 分割掩码已保存：{output_mask}")
    else:
        print("⚠️ 分割失败，未生成掩码")

    return mask


# ----------------------- 主程序入口 -----------------------
if __name__ == '__main__':
    seg_mask = segment_image('color_img_path.jpg')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
