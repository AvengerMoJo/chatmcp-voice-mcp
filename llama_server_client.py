#!/usr/bin/env python3
"""
MiniCPM-o 4.5 llama-server Python client
Tests the HTTP API for streaming audio interaction
"""

import requests
import json
import time
import os
import wave
import numpy as np

SERVER_URL = "http://localhost:9060"
MODEL_DIR = "/home/alex/.lmstudio/models/openbmb/MiniCPM-o-4_5-gguf"
OUTPUT_DIR = "/tmp/omni_output"

def wait_for_health(timeout=120):
    """Wait for server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                print(f"✓ Server healthy: {resp.json()}")
                return True
        except Exception as e:
            print(f"  Waiting for server... ({e})")
        time.sleep(2)
    return False

def omni_init():
    """Initialize omni session"""
    payload = {
        "media_type": 1,  # audio only
        "use_tts": False,  # disable for now
        "duplex_mode": True,
        "model_dir": MODEL_DIR,
        "tts_bin_dir": f"{MODEL_DIR}/tts",
        "tts_gpu_layers": 100,
        "token2wav_device": "gpu:0",
        "output_dir": OUTPUT_DIR,
        "voice_audio": f"{MODEL_DIR}/token2wav-gguf/prompt_cache.gguf"
    }

    print("\n📡 POST /v1/stream/omni_init")
    resp = requests.post(f"{SERVER_URL}/v1/stream/omni_init", json=payload, timeout=60)
    print(f"   Status: {resp.status_code}")
    result = resp.json()
    print(f"   Response: {json.dumps(result, indent=2)}")
    return result.get("success", False)

def stream_prefill(audio_path, cnt):
    """Send audio chunk"""
    payload = {
        "audio_path_prefix": audio_path,
        "img_path_prefix": "",
        "cnt": cnt
    }

    resp = requests.post(f"{SERVER_URL}/v1/stream/prefill", json=payload, timeout=30)
    return resp.json() if resp.status_code == 200 else None

def stream_decode(debug_dir=None):
    """Get streaming response"""
    payload = {
        "debug_dir": debug_dir or OUTPUT_DIR,
        "stream": True
    }

    resp = requests.post(f"{SERVER_URL}/v1/stream/decode", json=payload, stream=True, timeout=120)
    return resp

def run_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("MiniCPM-o 4.5 llama-server Client Test")
    print("=" * 60)

    if not wait_for_health():
        print("❌ Server not ready")
        return

    print("\n📡 Initializing omni session...")
    if not omni_init():
        print("❌ omni_init failed")
        return

    # Find test audio file
    test_audio = "/home/alex/Development/Sandbox/chatmcp/voice/llama.cpp-omni/tools/omni/assets/test_case/audio_test_case/audio_test_case_0001.wav"

    if not os.path.exists(test_audio):
        print(f"⚠️ Test audio not found: {test_audio}")
        print("   Creating synthetic test audio...")
        # Create 1 second of 16kHz audio
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save as WAV
        with wave.open(f"{OUTPUT_DIR}/test_input.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        test_audio = f"{OUTPUT_DIR}/test_input.wav"

    print(f"\n📡 Sending audio: {test_audio}")

    # Prefill
    result = stream_prefill(test_audio, cnt=1)
    if result:
        print(f"   Prefill result: {result}")

    # Decode with streaming
    print("\n📡 Starting decode stream...")
    resp = stream_decode()

    if resp.status_code == 200:
        print("   Streaming response:")
        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    data = json.loads(line[5:])
                    print(f"   >> {data}")
                elif line == 'data: [DONE]':
                    print("   ✓ Stream complete")
                    break
    else:
        print(f"   ❌ Decode failed: {resp.status_code} - {resp.text}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_test()