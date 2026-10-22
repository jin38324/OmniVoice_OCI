#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design via OpenAI-compatible API.

Usage:
    omnivoice-demo --port 8000
"""

import argparse
import base64
import io
import logging
import tempfile
import os
from typing import Any, Dict, Optional

import gradio as gr
import numpy as np
import soundfile as sf
from openai import OpenAI

# 语言映射数据（简化版，只保留demo需要的）
LANG_NAMES = {
    "zh": "Chinese (Mandarin)",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "fil": "Filipino",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "el": "Greek",
    "he": "Hebrew",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fa": "Persian",
    "am": "Amharic",
    "sw": "Swahili",
    "zu": "Zulu",
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "ne": "Nepali",
    "si": "Sinhala",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "ka": "Georgian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "uz": "Uzbek",
    "tg": "Tajik",
    "mn": "Mongolian",
    "bo": "Tibetan",
    "dsb": "Lower Sorbian",
    "hsb": "Upper Sorbian",
}

def lang_display_name(lang_code: str) -> str:
    """获取语言显示名称"""
    return LANG_NAMES.get(lang_code, lang_code)


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
_INSTRUCT_CATEGORIES = [
    {"male": "男", "female": "女"},
    {"child": "儿童", "teenager": "少年", "young adult": "青年",
     "middle-aged": "中年", "elderly": "老年"},
    {"very low pitch": "极低音调", "low pitch": "低音调",
     "moderate pitch": "中音调", "high pitch": "高音调",
     "very high pitch": "极高音调"},
    {"whisper": "耳语"},
    # Accent (English-only, no Chinese counterpart)
    {"american accent": "美式口音", "british accent": "英国口音", "australian accent": "澳大利亚口音",
     "chinese accent": "中国口音", "canadian accent": "加拿大口音", "indian accent": "印度口音",
     "korean accent": "韩语口音", "portuguese accent": "葡萄牙口音", "russian accent": "俄罗斯口音", "japanese accent": "日本口音"},
    # Dialect (Chinese-only, no English counterpart)
    {"河南话": "河南话", "陕西话": "陕西话", "四川话": "四川话", "贵州话": "贵州话", "云南话": "云南话", "桂林话": "桂林话",
     "济南话": "济南话", "石家庄话": "石家庄话", "甘肃话": "甘肃话", "宁夏话": "宁夏话", "青岛话": "青岛话", "东北话": "东北话"},
]

_ATTR_INFO = {
    "american accent": "Only effective for English speech.",
    "british accent": "Only effective for English speech.",
    "australian accent": "Only effective for English speech.",
    "chinese accent": "Only effective for English speech.",
    "canadian accent": "Only effective for English speech.",
    "indian accent": "Only effective for English speech.",
    "korean accent": "Only effective for English speech.",
    "portuguese accent": "Only effective for English speech.",
    "russian accent": "Only effective for English speech.",
    "japanese accent": "Only effective for English speech.",
    "河南话": "Only effective for Chinese speech.",
    "陕西话": "Only effective for Chinese speech.",
    "四川话": "Only effective for Chinese speech.",
    "贵州话": "Only effective for Chinese speech.",
    "云南话": "Only effective for Chinese speech.",
    "桂林话": "Only effective for Chinese speech.",
    "济南话": "Only effective for Chinese speech.",
    "石家庄话": "Only effective for Chinese speech.",
    "甘肃话": "Only effective for Chinese speech.",
    "宁夏话": "Only effective for Chinese speech.",
    "青岛话": "Only effective for Chinese speech.",
    "东北话": "Only effective for Chinese speech.",
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    return parser


# ---------------------------------------------------------------------------
# OpenAI API utilities
# ---------------------------------------------------------------------------


def numpy_to_base64(audio_array: np.ndarray, sample_rate: int = 24000) -> str:
    """将numpy音频数组转换为base64编码字符串"""
    # 如果audio_array是元组(sampling_rate, audio_data)，则提取音频数据
    if isinstance(audio_array, tuple) and len(audio_array) == 2:
        sr, audio_data = audio_array
        sample_rate = sr
        audio_array = audio_data

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    audio_data = buffer.read()
    return base64.b64encode(audio_data).decode("utf-8")


def call_openai_api(
    client: OpenAI,
    endpoint_id: str,
    text: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    speed: float = 1.0,
    mode: str = "auto",
) -> tuple:
    """调用OpenAI兼容的API进行语音生成"""
    try:
        extra_body = {"non_streaming_mode": True}

        # 根据模式设置不同的参数
        if mode == "clone" and ref_audio is not None:
            # 语音克隆模式 - 参考验证过的代码
            if isinstance(ref_audio, str):
                # 文件路径
                with open(ref_audio, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            elif isinstance(ref_audio, np.ndarray):
                # numpy数组 - 转换为WAV格式
                audio_b64 = numpy_to_base64(ref_audio)
            elif isinstance(ref_audio, tuple) and len(ref_audio) == 2:
                # (sampling_rate, audio_array) 元组格式
                sr, audio_array = ref_audio
                audio_b64 = numpy_to_base64(audio_array, sample_rate=sr)
            else:
                return None, f"Error: 不支持的音频格式: {type(ref_audio).__name__}"

            if audio_b64:
                extra_body["ref_audio"] = f"data:audio/wav;base64,{audio_b64}"
                if ref_text:
                    extra_body["ref_text"] = ref_text

        elif mode == "design" and instruct is not None and instruct:
            # 语音设计模式
            extra_body["task_type"] = "VoiceDesign"
            extra_body["instructions"] = instruct
            print(f"Debug: Voice Design mode, instructions = {instruct}")  # 调试信息

        # 构建API参数信息用于显示
        api_params = {
            "model": endpoint_id,
            "voice": None,
            "input": text,
            "response_format": "wav",
            "speed": speed,
            "extra_body": extra_body.copy()  # 复制一份避免修改原数据
        }

        # 处理ref_audio太长的问题，只保留头尾
        if "ref_audio" in api_params["extra_body"]:
            ref_audio = api_params["extra_body"]["ref_audio"]
            if isinstance(ref_audio, str) and len(ref_audio) > 100:
                # 保留前50个字符和后50个字符
                api_params["extra_body"]["ref_audio"] = ref_audio[:50] + "..." + ref_audio[-50:]

        # 格式化参数显示为JSON
        import json
        params_display = f"API Parameters:\n{json.dumps(api_params, indent=2, ensure_ascii=False)}"

        print(f"Debug: API Parameters:\n{params_display}")  # 调试信息

        # 调用API
        response = client.audio.speech.create(
            model=endpoint_id,
            voice=None,  # OmniVoice 无内置 voice preset
            input=text,
            response_format="wav",
            speed=speed,
            extra_body=extra_body,
        )

        # 保存到临时文件并读取
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            response.stream_to_file(temp_path)

        try:
            audio_array, sr = sf.read(temp_path)
            return (sr, audio_array), f"Done.\n{params_display}"
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        import traceback
        error_msg = f"Error: {type(e).__name__}: {str(e)}"
        # 添加详细的堆栈跟踪
        traceback_str = traceback.format_exc()
        if "BytesIO" in str(e):
            error_msg += " (音频格式错误，请确保上传的是有效的音频文件)"
        return None, f"{error_msg}\n详细信息:\n{traceback_str}"


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    # -- API配置状态 --
    api_config = {
        "base_url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1/",
        "api_key": "",
        "endpoint_id": "",
        "client": None,
    }

    def update_client(base_url: str, api_key: str, endpoint_id: str):
        """更新API客户端配置"""
        if not api_key or not endpoint_id:
            api_config["client"] = None
            return "❌ API Key 或 Endpoint ID 未设置"

        try:
            client = OpenAI(
                base_url=base_url.rstrip("/"),
                api_key=api_key
            )
            api_config["base_url"] = base_url
            api_config["api_key"] = api_key
            api_config["endpoint_id"] = endpoint_id
            api_config["client"] = client
            return "✅ API 配置成功"
        except Exception as e:
            api_config["client"] = None
            return f"❌ API 配置失败: {e}"

    # -- shared generation core --
    def _gen_core(
        text,
        ref_audio,
        instruct,
        speed,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        client = api_config["client"]
        if not client:
            return None, "❌ 请先配置 API (base_url, api_key, endpoint_id)"

        endpoint_id = api_config["endpoint_id"]

        return call_openai_api(
            client=client,
            endpoint_id=endpoint_id,
            text=text.strip(),
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct.strip() if isinstance(instruct, str) and instruct else None,
            speed=float(speed) if speed else 1.0,
            mode=mode,
        )

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    .api-config {background-color: #f0f8ff !important; border: 2px solid #4a90e2 !important; border-radius: 8px !important; padding: 16px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        with gr.Sidebar(width=600):
            gr.Markdown("""
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
            )

            # API配置区域
            with gr.Accordion("API Configuration / API 配置", open=True, elem_classes="api-config"):
                base_url = gr.Textbox(
                    label="Base URL",
                    value="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1/",
                    placeholder="API base URL",
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter your API key: sk-......",
                )
                endpoint_id = gr.Textbox(
                    label="Endpoint ID",
                    placeholder="Enter your endpoint ID: ocid1.generativeaiendpoint.oc1.<region>.......",
                )
                config_btn = gr.Button("Update Configuration / 更新配置", variant="primary")
                config_status = gr.Textbox(label="Status / 状态", interactive=False)

            config_btn.click(
                update_client,
                inputs=[base_url, api_key, endpoint_id],
                outputs=[config_status],
            )

        # 主内容区域
        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="numpy",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional) / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio.",
                        )
                        vc_speed = gr.Slider(
                            0.5,
                            2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed",
                            info="1.0 = normal. >1 faster, <1 slower.",
                        )
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(text, ref_aud, ref_text, speed):
                    # 处理Gradio Audio组件返回的格式
                    try:
                        if ref_aud is not None and isinstance(ref_aud, tuple) and len(ref_aud) == 2:
                            sr, audio_data = ref_aud
                            ref_aud = audio_data

                        return _gen_core(
                            text,
                            ref_aud,
                            None,  # instruct
                            speed,
                            mode="clone",
                            ref_text=ref_text or None,
                        )
                    except Exception as e:
                        return None, f"Error: 音频处理失败 - {type(e).__name__}: {str(e)}"

                vc_btn.click(
                    _clone_fn,
                    inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_speed],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )

                        _AUTO = "Auto"
                        vd_groups = []
                        for category_dict in _INSTRUCT_CATEGORIES:
                            # 将字典转换为选项列表
                            choices = [_AUTO] + [f"{en} / {zh}" for en, zh in category_dict.items()]
                            # 获取第一个键作为标签
                            label_key = list(category_dict.keys())[0]
                            # 根据内容设置合适的标签
                            if "male" in category_dict or "female" in category_dict:
                                label = "Gender / 性别"
                            elif "child" in category_dict or "teenager" in category_dict:
                                label = "Age / 年龄"
                            elif "pitch" in label_key:
                                label = "Pitch / 音调"
                            elif "whisper" in category_dict:
                                label = "Style / 风格"
                            elif "accent" in label_key:
                                label = "English Accent / 英文口音"
                            else:
                                label = "Chinese Dialect / 中文方言"

                            # 设置info信息
                            info = None
                            for key in category_dict.keys():
                                if key in _ATTR_INFO:
                                    info = _ATTR_INFO[key]
                                    break

                            vd_groups.append(
                                gr.Dropdown(
                                    label=label,
                                    choices=choices,
                                    value=_AUTO,
                                    info=info,
                                )
                            )

                        vd_speed = gr.Slider(
                            0.5,
                            2.0,
                            value=1.0,
                            step=0.1,
                            label="Speed",
                            info="1.0 = normal. >1 faster, <1 slower.",
                        )
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns."""
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # 检查是否是中文方言（没有英文部分）
                            if en == zh:  # 如果两部分相同，说明是纯中文
                                parts.append(zh.strip())
                            else:
                                # 英文部分已经是小写，直接使用
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    result = ", ".join(parts)
                    print(f"Debug: _build_instruct result = {result}")  # 调试信息
                    return result

                def _design_fn(text, speed, *groups):
                    instruct = _build_instruct(groups)
                    print(f"Debug: _design_fn instruct = {instruct}")  # 调试信息
                    return _gen_core(
                        text,
                        None,  # ref_audio
                        instruct,
                        speed,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[vd_text, vd_speed] + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    demo = build_demo()

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
