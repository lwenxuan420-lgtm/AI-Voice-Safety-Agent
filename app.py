import os
import re
import json
import time
import uuid
import shutil
import random
import secrets
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import torchaudio
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from huggingface_hub import InferenceClient
from faster_whisper import WhisperModel

from model import CNN2D

# =========================================================
# GemmaShield - Hugging Face Runnable Full MVP
# Features:
# - Phone login + demo SMS verification code
# - Elder/helper account roles
# - Binding code + elder confirmation: helper applies, elder approves, elder binds up to 5 trusted helpers
# - Simulated SMS / WeChat Service Account / App Push notification layer
# - Local JSON database for HF Space demo, designed to migrate to Supabase/Firebase/PostgreSQL
# - Permission control: helper only sees requests from bound/notified elders
# - Unbind / replace trusted contacts
# - Two-stage analysis: CNN + Whisper first, Gemma 4 API second
# =========================================================

APP_NAME = "GemmaShield"
SR = 16000
LENGTH = 64000
MAX_HELPERS_PER_ELDER = 5

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "requests_db.json"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
GEMMA_MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-26B-A4B-it")
GEMMA_PROVIDER = os.getenv("GEMMA_PROVIDER", "deepinfra")

# Notification provider secrets are optional. In HF demo mode, notifications are simulated.
SMS_PROVIDER_ENABLED = bool(os.getenv("SMS_API_KEY"))
WECHAT_PROVIDER_ENABLED = bool(os.getenv("WECHAT_APP_ID") and os.getenv("WECHAT_APP_SECRET"))
PUSH_PROVIDER_ENABLED = bool(os.getenv("PUSH_API_KEY"))

# Keep CPU stable on Hugging Face.
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))

app = FastAPI(title=APP_NAME)

# =========================================================
# Internationalization (Chinese / English)
# =========================================================

SUPPORTED_LANGS = {"zh", "en"}

def normalize_lang(lang: str) -> str:
    """Normalize UI / reasoning language. Chinese is the default."""
    lang = str(lang or "zh").strip().lower()
    if lang in {"en", "en-us", "en_us", "english"}:
        return "en"
    return "zh"


def localized(lang: str, zh: str, en: str) -> str:
    return en if normalize_lang(lang) == "en" else zh

# Lazy model holders. This keeps /, /elder, /helper alive even before model loading.
_cnn = None
_mel = None
_whisper = None
_gemma_client = None

# =========================================================
# Database layer
# =========================================================
# For Hugging Face MVP, we use local JSON persistence.
# Real deployment can replace this layer with Supabase/Firebase/PostgreSQL.
# The rest of the app uses load_db()/save_db(), so migration is isolated.

DEFAULT_DB = {
    "users": {},          # user_id -> user profile
    "phone_index": {},    # normalized phone -> user_id
    "otps": {},           # normalized phone -> otp record
    "requests": {},       # request_id -> request
    "notifications": {},  # notification_id -> simulated notification
    "binding_requests": {},  # binding_request_id -> pending/approved/rejected helper binding request
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def normalize_phone(phone: str) -> str:
    phone = str(phone or "").strip()
    phone = re.sub(r"[^0-9+]", "", phone)
    return phone


def unique_list(values: List[str]) -> List[str]:
    """Keep list order while removing duplicates and empty values."""
    seen = set()
    out = []
    for v in values or []:
        if not v:
            continue
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def ensure_db_shape(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and repair local JSON DB.

    Important fix:
    Earlier versions could show an elder as bound on the elder side, while the helper side
    still displayed 0 bound elders. The reason was that binding relations may exist in
    elder.bound_helpers or old request.notified_helpers but not in helper.bound_elders.
    This function repairs the two-way relation every time the DB is loaded/saved.
    """
    if not isinstance(data, dict):
        data = {}
    for k, v in DEFAULT_DB.items():
        if k not in data or not isinstance(data[k], type(v)):
            data[k] = v.copy() if isinstance(v, dict) else v

    users = data.get("users", {})

    # 1) Migrate users that may miss fields.
    for uid, user in list(users.items()):
        if not isinstance(user, dict):
            users.pop(uid, None)
            continue
        user.setdefault("user_id", uid)
        user.setdefault("phone", "")
        user.setdefault("role", "helper")
        user.setdefault("display_name", "用户")
        user.setdefault("created_at", now_str())
        user.setdefault("session_tokens", [])
        if user.get("role") == "elder":
            # Elder-facing code is always six digits. Older letter-based demo codes
            # are migrated automatically so the elder only needs to read numbers.
            current_bind_code = str(user.get("bind_code") or "").strip()
            if not re.fullmatch(r"\d{6}", current_bind_code):
                user["bind_code"] = make_bind_code(uid)
            user["bound_helpers"] = unique_list(user.get("bound_helpers", []))
        if user.get("role") == "helper":
            user["bound_elders"] = unique_list(user.get("bound_elders", []))
            user.setdefault("relationship", "family")
            user.setdefault("channels", ["sms", "wechat", "push"])

    # 2) Rebuild phone index.
    data["phone_index"] = {}
    for uid, user in users.items():
        phone = normalize_phone(user.get("phone", ""))
        if phone:
            data["phone_index"][phone] = uid

    # 3) Repair two-way binding from elders -> helpers.
    for elder_id, elder in list(users.items()):
        if elder.get("role") != "elder":
            continue
        repaired_helpers = []
        for helper_id in unique_list(elder.get("bound_helpers", [])):
            helper = users.get(helper_id)
            if not helper or helper.get("role") != "helper":
                continue
            repaired_helpers.append(helper_id)
            helper.setdefault("bound_elders", [])
            if elder_id not in helper["bound_elders"]:
                helper["bound_elders"].append(elder_id)
        elder["bound_helpers"] = unique_list(repaired_helpers)

    # 4) Repair two-way binding from helpers -> elders.
    for helper_id, helper in list(users.items()):
        if helper.get("role") != "helper":
            continue
        repaired_elders = []
        for elder_id in unique_list(helper.get("bound_elders", [])):
            elder = users.get(elder_id)
            if not elder or elder.get("role") != "elder":
                continue
            repaired_elders.append(elder_id)
            elder.setdefault("bound_helpers", [])
            if helper_id not in elder["bound_helpers"] and len(elder["bound_helpers"]) < MAX_HELPERS_PER_ELDER:
                elder["bound_helpers"].append(helper_id)
        helper["bound_elders"] = unique_list(repaired_elders)

    # 5) Repair from historical requests.
    # If a helper received a request from an elder, the UI should treat them as related
    # for this MVP so that helper profile does not show 0 while inbox has requests.
    for req in data.get("requests", {}).values():
        if not isinstance(req, dict):
            continue
        elder_id = req.get("elder_id")
        elder = users.get(elder_id)
        if not elder or elder.get("role") != "elder":
            continue
        for helper_id in unique_list(req.get("notified_helpers", [])):
            helper = users.get(helper_id)
            if not helper or helper.get("role") != "helper":
                continue
            helper.setdefault("bound_elders", [])
            if elder_id not in helper["bound_elders"]:
                helper["bound_elders"].append(elder_id)
            elder.setdefault("bound_helpers", [])
            if helper_id not in elder["bound_helpers"] and len(elder["bound_helpers"]) < MAX_HELPERS_PER_ELDER:
                elder["bound_helpers"].append(helper_id)

    # 6) Deduplicate again.
    for uid, user in users.items():
        if user.get("role") == "elder":
            user["bound_helpers"] = unique_list(user.get("bound_helpers", []))[:MAX_HELPERS_PER_ELDER]
        if user.get("role") == "helper":
            user["bound_elders"] = unique_list(user.get("bound_elders", []))
            user["channels"] = unique_list(user.get("channels", ["sms", "wechat", "push"])) or ["sms", "wechat", "push"]

    # 7) Normalize pending binding requests.
    for bid, br in list(data.get("binding_requests", {}).items()):
        if not isinstance(br, dict):
            data["binding_requests"].pop(bid, None)
            continue
        br.setdefault("binding_request_id", bid)
        br.setdefault("status", "pending")
        br.setdefault("created_at", now_str())
        br.setdefault("decided_at", None)

    return data

def load_db() -> Dict[str, Any]:
    if not DB_PATH.exists():
        save_db(DEFAULT_DB.copy())
    try:
        data = json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = DEFAULT_DB.copy()
    return ensure_db_shape(data)


def save_db(data: Dict[str, Any]) -> None:
    data = ensure_db_shape(data)
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DB_PATH)


def make_bind_code(user_id: str) -> str:
    """Create a stable six-digit demo code that is easy for elders to read aloud."""
    clean = re.sub(r"[^A-F0-9]", "", str(user_id).upper())
    seed = int((clean[-12:] or "0"), 16)
    return f"{seed % 900000 + 100000:06d}"



def user_public(user: Dict[str, Any], db: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    db = ensure_db_shape(db or load_db())
    user_id = user.get("user_id")

    result = {
        "user_id": user_id,
        "phone": user.get("phone"),
        "role": user.get("role"),
        "display_name": user.get("display_name"),
        "created_at": user.get("created_at"),
    }

    if user.get("role") == "elder":
        helper_ids = list(user.get("bound_helpers", []))
        # Extra safety: include helpers that have this elder in reverse relation.
        for hid, h in db.get("users", {}).items():
            if h.get("role") == "helper" and user_id in h.get("bound_elders", []):
                helper_ids.append(hid)
        helper_ids = unique_list(helper_ids)[:MAX_HELPERS_PER_ELDER]

        helpers = []
        for hid in helper_ids:
            h = db["users"].get(hid)
            if h:
                helpers.append({
                    "user_id": hid,
                    "display_name": h.get("display_name"),
                    "phone": mask_phone(h.get("phone")),
                    "relationship": h.get("relationship", "family"),
                    "channels": h.get("channels", []),
                })
        pending_bindings = []
        for bid, br in db.get("binding_requests", {}).items():
            if br.get("elder_id") == user_id and br.get("status") == "pending":
                h = db.get("users", {}).get(br.get("helper_id"), {})
                pending_bindings.append({
                    "binding_request_id": bid,
                    "helper_id": br.get("helper_id"),
                    "helper_name": br.get("helper_name") or h.get("display_name", "协助人"),
                    "helper_phone": mask_phone(br.get("helper_phone") or h.get("phone", "")),
                    "relationship": br.get("relationship") or h.get("relationship", "family"),
                    "created_at": br.get("created_at"),
                })

        result.update({
            "bind_code": user.get("bind_code"),
            "bound_helpers": helpers,
            "helper_count": len(helpers),
            "max_helpers": MAX_HELPERS_PER_ELDER,
            "pending_binding_requests": pending_bindings,
        })

    if user.get("role") == "helper":
        elder_ids = list(user.get("bound_elders", []))
        # Extra safety: include elders that contain this helper in bound_helpers.
        for eid, e in db.get("users", {}).items():
            if e.get("role") == "elder" and user_id in e.get("bound_helpers", []):
                elder_ids.append(eid)
        # Extra safety: include elders from historical inbox requests.
        for req in db.get("requests", {}).values():
            if isinstance(req, dict) and user_id in req.get("notified_helpers", []):
                if req.get("elder_id"):
                    elder_ids.append(req.get("elder_id"))
        elder_ids = unique_list(elder_ids)

        elders = []
        for eid in elder_ids:
            e = db["users"].get(eid)
            if e:
                elders.append({
                    "user_id": eid,
                    "display_name": e.get("display_name"),
                    "phone": mask_phone(e.get("phone")),
                    "bind_code": e.get("bind_code"),
                })
        pending_bindings = []
        for bid, br in db.get("binding_requests", {}).items():
            if br.get("helper_id") == user_id and br.get("status") == "pending":
                e = db.get("users", {}).get(br.get("elder_id"), {})
                pending_bindings.append({
                    "binding_request_id": bid,
                    "elder_id": br.get("elder_id"),
                    "elder_name": br.get("elder_name") or e.get("display_name", "老人"),
                    "elder_phone": mask_phone(e.get("phone", "")),
                    "created_at": br.get("created_at"),
                })

        result.update({
            "bound_elders": elders,
            "relationship": user.get("relationship", "family"),
            "channels": user.get("channels", []),
            "pending_binding_requests": pending_bindings,
        })
    return result

def mask_phone(phone: str) -> str:
    p = normalize_phone(phone)
    if len(p) <= 4:
        return p
    return p[:3] + "****" + p[-4:]


def auth_user(token: str, role: Optional[str] = None) -> Dict[str, Any]:
    token = str(token or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    db = load_db()
    for user in db["users"].values():
        if token in user.get("session_tokens", []):
            if role and user.get("role") != role:
                raise HTTPException(status_code=403, detail=f"Requires role: {role}")
            return user
    raise HTTPException(status_code=401, detail="Invalid or expired token")


def can_access_request(user: Dict[str, Any], req: Dict[str, Any]) -> bool:
    if user.get("role") == "elder":
        return req.get("elder_id") == user.get("user_id")
    if user.get("role") == "helper":
        return user.get("user_id") in req.get("notified_helpers", [])
    return False

# =========================================================
# Notification layer
# =========================================================
# In HF Space, real SMS/WeChat/Push is not connected.
# These functions simulate delivery and keep logs in requests_db.json.
# To deploy for real, replace these functions with Aliyun/Twilio SMS,
# WeChat Service Account API, Firebase/APNs/FCM App Push, etc.


def create_notification(db: Dict[str, Any], request_id: str, elder: Dict[str, Any], helper: Dict[str, Any], channel: str) -> Dict[str, Any]:
    nid = new_id("N")
    provider_enabled = {
        "sms": SMS_PROVIDER_ENABLED,
        "wechat": WECHAT_PROVIDER_ENABLED,
        "push": PUSH_PROVIDER_ENABLED,
    }.get(channel, False)

    # Demo mode: do not send real messages. Store simulated delivery.
    notification = {
        "notification_id": nid,
        "request_id": request_id,
        "elder_id": elder.get("user_id"),
        "elder_name": elder.get("display_name"),
        "helper_id": helper.get("user_id"),
        "helper_name": helper.get("display_name"),
        "channel": channel,
        "provider_enabled": provider_enabled,
        "status": "simulated_delivered" if not provider_enabled else "provider_hook_ready",
        "message": f"{elder.get('display_name')} 发起了 GemmaShield 求助请求 {request_id}",
        "created_at": now_str(),
    }
    db["notifications"][nid] = notification
    return notification


def dispatch_notifications(db: Dict[str, Any], request_id: str, elder: Dict[str, Any], helper_ids: List[str]) -> List[Dict[str, Any]]:
    sent = []
    for hid in helper_ids:
        helper = db["users"].get(hid)
        if not helper:
            continue
        channels = helper.get("channels") or ["sms", "wechat", "push"]
        for ch in channels:
            if ch in ["sms", "wechat", "push"]:
                sent.append(create_notification(db, request_id, elder, helper, ch))
    return sent

# =========================================================
# Model loading and audio preprocessing
# =========================================================


def get_cnn():
    global _cnn
    if _cnn is None:
        model_path = BASE_DIR / "best_model.pth"
        if not model_path.exists():
            raise RuntimeError("best_model.pth not found. Please keep your original CNN checkpoint file.")
        print("Loading CNN model...")
        model = CNN2D()
        state = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state)
        model.to("cpu").eval()
        _cnn = model
        print("CNN ready.")
    return _cnn


def get_mel():
    global _mel
    if _mel is None:
        _mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=1024,
            hop_length=512,
            n_mels=64,
        )
    return _mel


def get_whisper():
    global _whisper
    if _whisper is None:
        print("Loading Whisper tiny...")
        _whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Whisper ready.")
    return _whisper


def get_gemma_client():
    global _gemma_client
    if _gemma_client is not None:
        return _gemma_client
    if not HF_TOKEN:
        return None
    try:
        print(f"Connecting Gemma 4 API: {GEMMA_MODEL_ID}, provider={GEMMA_PROVIDER}")
        _gemma_client = InferenceClient(
            provider=GEMMA_PROVIDER,
            model=GEMMA_MODEL_ID,
            token=HF_TOKEN,
            timeout=60,
        )
        return _gemma_client
    except Exception as e:
        print("Gemma client init failed:", repr(e))
        _gemma_client = None
        return None


def norm_audio(path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", str(SR), "-ac", "1", tmp.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print("ffmpeg conversion failed; using original path")
        return path
    return tmp.name


def load_audio(path: str) -> torch.Tensor:
    wav, sr = sf.read(path)
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav).float()
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    wav = (wav - wav.mean()) / (wav.std() + 1e-8)
    if wav.shape[0] < LENGTH:
        wav = torch.nn.functional.pad(wav, (0, LENGTH - wav.shape[0]))
    else:
        wav = wav[:LENGTH]
    return wav


def make_feat(wav: torch.Tensor) -> torch.Tensor:
    mel = get_mel()(wav)
    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel.unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def detect_audio(audio_path: str) -> Dict[str, float]:
    wav = load_audio(audio_path)
    x = make_feat(wav)
    out = get_cnn()(x).squeeze()
    real = torch.sigmoid(out).item()
    fake = 1 - real
    return {"fake": float(fake), "real": float(real)}


def transcribe_audio(audio_path: str) -> str:
    try:
        segments, _ = get_whisper().transcribe(audio_path)
        text = " ".join([s.text for s in segments]).strip()
        return text if text else "No clear speech detected."
    except Exception as e:
        print("Whisper transcribe failed:", repr(e))
        return "No clear speech detected."

# =========================================================
# Risk logic
# =========================================================


def level_icon(level: str) -> str:
    return {"HIGH": "🚨", "MEDIUM": "⚠️", "VERIFY": "🟡", "LOW": "✅"}.get(level, "⚠️")


def level_label(level: str, lang: str = "zh") -> str:
    if lang == "zh":
        return {"HIGH": "高风险", "MEDIUM": "谨慎", "VERIFY": "需核实", "LOW": "相对安全"}.get(level, level)
    return {"HIGH": "HIGH RISK", "MEDIUM": "CAUTION", "VERIFY": "NEEDS VERIFICATION", "LOW": "RELATIVELY SAFE"}.get(level, level)


def scam_type(text: str, fake: float) -> str:
    t = (text or "").lower()
    if fake >= 0.78:
        return "AI voice spoofing risk"
    if any(x in t for x in ["investment", "crypto", "profit", "money", "bank", "transfer", "投资", "赚钱", "转账", "银行卡", "收益", "理财"]):
        return "financial or investment scam"
    if any(x in t for x in ["mom", "dad", "grandma", "grandpa", "urgent", "hospital", "help", "妈妈", "爸爸", "爷爷", "奶奶", "外公", "外婆", "紧急", "医院", "救我"]):
        return "family impersonation scam"
    if any(x in t for x in ["verification code", "otp", "password", "code", "验证码", "密码", "短信", "账号"]):
        return "verification code scam"
    return "unknown or unclear risk"


def scam_label(scam: str, lang: str = "zh") -> str:
    zh = {
        "AI voice spoofing risk": "AI 语音伪造风险",
        "financial or investment scam": "金融或投资诈骗风险",
        "family impersonation scam": "冒充亲属诈骗风险",
        "verification code scam": "验证码诈骗风险",
        "unknown or unclear risk": "未知或不明确风险",
    }
    return zh.get(scam, scam) if lang == "zh" else scam


def extract_risk_signals(text: str) -> List[str]:
    t = (text or "").lower()
    checks = [
        ("money_transfer", ["transfer", "bank", "money", "account", "转账", "银行卡", "账号", "钱"]),
        ("verification_code", ["verification", "otp", "code", "password", "验证码", "密码", "短信"]),
        ("urgency_pressure", ["urgent", "immediately", "right now", "紧急", "马上", "立刻", "现在"]),
        ("family_impersonation", ["mom", "dad", "grandma", "grandpa", "妈妈", "爸爸", "爷爷", "奶奶"]),
        ("investment_profit", ["investment", "crypto", "profit", "投资", "理财", "收益", "赚钱"]),
    ]
    signals = [name for name, words in checks if any(w in t for w in words)]
    return signals if signals else ["no obvious keyword signal"]


def signal_label(signal: str, lang: str = "zh") -> str:
    zh = {
        "money_transfer": "转账/资金相关信号",
        "verification_code": "验证码/密码相关信号",
        "urgency_pressure": "紧急施压信号",
        "family_impersonation": "冒充亲属信号",
        "investment_profit": "投资收益诱导信号",
        "no obvious keyword signal": "未发现明显关键词风险信号",
    }
    return zh.get(signal, signal) if lang == "zh" else signal.replace("_", " ")


def format_signals(signals: List[str], lang: str = "zh") -> str:
    return ", ".join(signal_label(s, lang) for s in signals)


# Evidence source matters because the CNN is strong on clean/direct audio,
# while microphone re-recording, playback devices, room noise and app compression
# may reduce acoustic reliability. We surface this as a product-level confidence note
# instead of hiding the limitation or over-explaining it to elderly users.
EVIDENCE_SOURCE_META = {
    "elder_recording": {
        "label_zh": "老人端现场录音",
        "label_en": "Older-adult microphone recording",
        "confidence": "review",
        "confidence_zh": "需结合人工核实",
        "confidence_en": "Human verification required",
        "note_zh": "该证据来自浏览器/麦克风现场保留，适合快速求助，但可能受到播放设备、距离和环境噪声影响，不建议仅凭声学分数判断。",
        "note_en": "This evidence was captured through a browser or microphone. It is useful for rapid assistance, but playback devices, distance, room acoustics, and background noise may reduce acoustic reliability. Do not rely on the acoustic score alone.",
    },
    "elder_upload": {
        "label_zh": "老人端上传音频",
        "label_en": "Older-adult uploaded audio",
        "confidence": "medium",
        "confidence_zh": "中等",
        "confidence_en": "Medium",
        "note_zh": "该证据由老人端上传，通常比现场录音更稳定，但仍建议结合通话内容和家属核实。",
        "note_en": "This evidence was uploaded by the older adult and is usually more stable than a live microphone re-recording. It should still be reviewed together with the conversation content and trusted-human verification.",
    },
    "wechat_voice": {
        "label_zh": "微信/社交软件语音",
        "label_en": "Social-media voice message",
        "confidence": "medium",
        "confidence_zh": "中等",
        "confidence_en": "Medium",
        "note_zh": "该证据可能经过社交软件压缩，适合判断话术风险，但声学判断仍需谨慎。",
        "note_en": "This evidence may have been compressed by a social-media platform. It is useful for semantic risk analysis, but acoustic conclusions should be treated cautiously.",
    },
    "voicemail": {
        "label_zh": "语音留言",
        "label_en": "Voicemail",
        "confidence": "medium",
        "confidence_zh": "中等",
        "confidence_en": "Medium",
        "note_zh": "该证据来自语音留言，适合转录和风险话术分析。",
        "note_en": "This evidence comes from voicemail and is suitable for transcription and scam-language analysis, while channel effects should still be considered.",
    },
    "call_recording": {
        "label_zh": "保存的通话录音",
        "label_en": "Saved call recording",
        "confidence": "high",
        "confidence_zh": "较高",
        "confidence_en": "Relatively high",
        "note_zh": "该证据来自已保存通话录音，通常比现场麦克风重录更适合声学分析。",
        "note_en": "This evidence comes from a saved call recording and is generally more suitable for acoustic analysis than a live microphone re-recording, although telephone-channel effects remain.",
    },
    "helper_upload": {
        "label_zh": "家属/社区端上传音频",
        "label_en": "Helper-uploaded audio",
        "confidence": "high",
        "confidence_zh": "较高",
        "confidence_en": "Relatively high",
        "note_zh": "该证据由家属/社区端补充上传，适合进行 CNN + Whisper 分析。",
        "note_en": "This evidence was uploaded by a trusted family or community helper and is suitable for CNN + Whisper analysis, subject to the quality of the original source.",
    },
}


def normalize_evidence_source(source: str) -> str:
    source = str(source or "").strip()
    aliases = {
        "elder_evidence": "elder_recording",
        "browser_recording": "elder_recording",
        "mic_recording": "elder_recording",
        "social_voice": "wechat_voice",
        "wechat": "wechat_voice",
        "call": "call_recording",
        "upload": "helper_upload",
    }
    source = aliases.get(source, source)
    if source not in EVIDENCE_SOURCE_META:
        return "helper_upload"
    return source


def evidence_meta(source: str) -> Dict[str, str]:
    return EVIDENCE_SOURCE_META[normalize_evidence_source(source)]


def risk_level(fake: float, text: str, audio_source: str = "helper_upload") -> str:
    t = (text or "").lower()
    risky_words = [
        "transfer", "bank", "password", "verification", "otp", "urgent",
        "money", "account", "crypto", "investment",
        "转账", "银行卡", "验证码", "密码", "紧急", "投资", "赚钱", "账号",
        "不要告诉", "别告诉", "保密", "马上", "立刻", "安全账户", "公检法"
    ]
    keyword_hit = any(w in t for w in risky_words)
    meta = evidence_meta(audio_source)
    confidence = meta.get("confidence", "medium")

    # CNN remains an acoustic evidence module. For clean/direct audio it can be useful,
    # especially when it strongly detects spoofing. When fake probability is low but
    # evidence comes from a microphone or compressed social source, we do not overstate
    # safety; we mark it as verification-needed.
    if fake >= 0.80 or (fake >= 0.55 and keyword_hit):
        return "HIGH"
    if fake >= 0.35 or keyword_hit:
        return "MEDIUM"
    if confidence == "review":
        return "VERIFY"
    return "LOW"


def quick_scan_audio(audio_path: str, audio_source: str = "helper_upload") -> Dict[str, Any]:
    start = time.time()
    source = normalize_evidence_source(audio_source)
    meta = evidence_meta(source)
    normalized = norm_audio(audio_path)
    probs = detect_audio(normalized)
    text = transcribe_audio(normalized)
    fake = probs["fake"]
    real = probs["real"]
    level = risk_level(fake, text, source)
    scam = scam_type(text, fake)
    signals = extract_risk_signals(text)

    # Store both languages so a saved request can be viewed after the user
    # switches UI language without re-running CNN/Whisper.
    return {
        "fake": fake,
        "real": real,
        "transcript": text,
        "level": level,
        "level_label_zh": level_label(level, "zh"),
        "level_label_en": level_label(level, "en"),
        "scam": scam,
        "scam_label_zh": scam_label(scam, "zh"),
        "scam_label_en": scam_label(scam, "en"),
        "signals": signals,
        "signals_label_zh": format_signals(signals, "zh"),
        "signals_label_en": format_signals(signals, "en"),
        "evidence_source": source,
        "evidence_source_label_zh": meta.get("label_zh"),
        "evidence_source_label_en": meta.get("label_en"),
        "evidence_confidence": meta.get("confidence"),
        "evidence_confidence_zh": meta.get("confidence_zh"),
        "evidence_confidence_en": meta.get("confidence_en"),
        "evidence_note_zh": meta.get("note_zh"),
        "evidence_note_en": meta.get("note_en"),
        "processing_seconds": round(time.time() - start, 2),
        "scanned_at": now_str(),
    }


def build_gemma_prompt(
    quick: Dict[str, Any],
    elder_name: str = "老人",
    lang: str = "zh",
) -> str:
    lang = normalize_lang(lang)
    fake = quick.get("fake", 0.0)
    real = quick.get("real", 0.0)
    text = quick.get("transcript", "")
    level = quick.get("level", "MEDIUM")
    scam = quick.get("scam", "unknown or unclear risk")
    signals = quick.get("signals", [])

    if lang == "en":
        source_label = quick.get("evidence_source_label_en") or quick.get("evidence_source_label_zh") or "Unknown source"
        confidence_label = quick.get("evidence_confidence_en") or "Medium"
        evidence_note = quick.get("evidence_note_en") or "Please verify the caller's identity with a trusted person."

        return f"""
You are Gemma 4, the risk-reasoning assistant in a voice-scam safety system designed to support older adults.

Respond only in English.
Keep the explanation concise, clear, calm, and easy to understand.
Do not repeat the full transcript.
Do not say "I am an AI model."
Do not mention APIs, fallback logic, or service availability.
Do not exaggerate system capabilities and do not claim that the system is connected to a real telephone network.

Important positioning:
- The CNN provides supporting acoustic evidence only; it is not the final judge.
- Whisper provides transcript evidence.
- Gemma 4 combines acoustic evidence, transcript evidence, scam-behavior signals, and source reliability to produce a risk explanation and practical advice.
- Do not describe Gemma 4 as "correcting" the CNN.
- If the evidence is a live microphone re-recording or compressed social-media audio, remind the helper to verify identity through an independent trusted channel. Do not label the call "safe" based on one acoustic score.

Input:
- Older adult: {elder_name}
- Real-speech probability: {real:.2f}
- AI-spoof probability: {fake:.2f}
- Detector risk level: {level_label(level, "en")}
- Evidence source: {source_label}
- Evidence confidence: {confidence_label}
- Evidence-use note: {evidence_note}
- Risk-type hint: {scam_label(scam, "en")}
- Risk signals: {format_signals(signals, "en")}
- Original transcript: {text[:450]}

Return exactly this structure:

1. Risk Level:
2. Why This May Be Risky:
3. Acoustic Evidence:
4. Textual Evidence:
5. Advice for the Older Adult:
6. Advice for the Family / Community Helper:
7. One-Sentence Warning:
""".strip()

    source_label = quick.get("evidence_source_label_zh") or "未知来源"
    confidence_label = quick.get("evidence_confidence_zh") or "中等"
    evidence_note = quick.get("evidence_note_zh") or "请结合家属核实。"

    return f"""
你是 Gemma 4，一个面向老年人 AI 语音诈骗防护系统的风险推理助手。

请只用中文回答。
请简洁、清楚、老人友好。
不要重复完整转录文本。
不要说“我是 AI 模型”。
不要提到 API、fallback、服务不可用。
不要夸大系统能力，不要说系统已经接入真实电话系统。

重要定位：
CNN 只提供声音层面的辅助证据，不是最终裁判。
Whisper 提供通话文本证据。
Gemma 4 负责综合声音证据、文本证据、诈骗行为信号和证据来源可靠性，生成风险解释和行动建议。
不要把 Gemma 4 表述为“修正 CNN”。
如果音频来自现场麦克风录音或社交软件压缩来源，请提醒家属结合独立渠道进行身份核实；不要直接用“安全”盖棺定论。

输入信息：
- 老人：{elder_name}
- 真实语音概率：{real:.2f}
- AI 伪造语音概率：{fake:.2f}
- 检测器风险等级：{level_label(level, "zh")}
- 证据来源：{source_label}
- 证据置信度：{confidence_label}
- 证据使用提醒：{evidence_note}
- 风险类型提示：{scam_label(scam, "zh")}
- 风险信号：{format_signals(signals, "zh")}
- 原始语音转录：{text[:450]}

请严格按下面格式输出：

1. 风险等级：
2. 为什么有风险：
3. 声音证据：
4. 文本证据：
5. 给老人的建议：
6. 给家属/社区的处理建议：
7. 一句话提醒：
""".strip()


def local_safety_reasoning(
    quick: Dict[str, Any],
    elder_name: str = "老人",
    lang: str = "zh",
) -> str:
    lang = normalize_lang(lang)
    fake = quick.get("fake", 0.0)
    real = quick.get("real", 0.0)
    text = quick.get("transcript", "")
    level = quick.get("level", "MEDIUM")
    scam = quick.get("scam", "unknown or unclear risk")
    signals = quick.get("signals", [])

    if lang == "en":
        source_label = quick.get("evidence_source_label_en") or "Unknown source"
        confidence_label = quick.get("evidence_confidence_en") or "Medium"
        evidence_note = quick.get("evidence_note_en") or "Verify the caller's identity through an independent trusted channel."

        if level == "HIGH":
            why = "The call contains strong risk indicators. The acoustic evidence and/or transcript includes signs such as synthetic speech, money-transfer requests, verification codes, or urgent pressure."
            elder_advice = "Stop following the caller's instructions. Do not transfer money or provide verification codes, bank details, passwords, or identity information. End the call and contact a trusted family or community helper."
            helper_advice = "Verify the situation using an official phone number, the bank's official channel, or an independent trusted contact. Do not simply call back the suspicious number."
            warning = "This call may be high risk. Stop the requested action and verify it with someone you trust."
        elif level == "MEDIUM":
            why = "The call contains some risk. The acoustic evidence is uncertain and/or the transcript contains suspicious signals that require further verification."
            elder_advice = "Do not immediately trust the caller, transfer money, or provide personal information. Ask a trusted helper to verify the situation first."
            helper_advice = "Review the transcript and independently verify the caller's identity, purpose, and any request involving money or verification codes."
            warning = "Verify the caller's identity before continuing."
        elif level == "VERIFY":
            why = "No strong high-risk phrase was detected and the acoustic score does not show strong spoof evidence, but the evidence source is not reliable enough to justify a safe conclusion."
            elder_advice = "Do not transfer money or provide verification codes until a trusted family or community helper confirms the caller's identity."
            helper_advice = "Check the caller's number, contact the real family member through a known channel, and obtain a clearer original recording if possible."
            warning = "The available evidence needs human verification before the call can be trusted."
        else:
            why = "The current audio has a relatively low spoof probability and the transcript does not contain an obvious money-transfer, verification-code, or urgent-pressure signal."
            elder_advice = "The current risk appears lower, but stop and verify immediately if the caller later asks for money, verification codes, bank information, or passwords."
            helper_advice = "Keep the record and remind the older adult that any request involving money or verification codes requires a second independent confirmation."
            warning = "Even when current risk appears lower, always verify requests involving money or verification codes."

        return f"""
1. Risk Level: {level_label(level, "en")}
2. Why This May Be Risky: {why}
3. Acoustic Evidence: CNN estimated AI-spoof probability {fake:.2f} and real-speech probability {real:.2f}. This score is supporting acoustic evidence only. Evidence source: "{source_label}". Evidence confidence: "{confidence_label}". {evidence_note}
4. Textual Evidence: Risk type: "{scam_label(scam, "en")}"; risk signals: "{format_signals(signals, "en")}". Original transcript excerpt: {text[:160]}
5. Advice for the Older Adult: {elder_advice}
6. Advice for the Family / Community Helper: {helper_advice}
7. One-Sentence Warning: {warning}
""".strip()

    source_label = quick.get("evidence_source_label_zh") or "未知来源"
    confidence_label = quick.get("evidence_confidence_zh") or "中等"
    evidence_note = quick.get("evidence_note_zh") or "请结合家属核实。"

    if level == "HIGH":
        why = "该通话存在明显风险：声音证据或文本内容中出现了 AI 伪造、转账、验证码、紧急施压等高危信号。"
        elder_advice = "请立即停止按照对方要求操作，不要转账，不要提供验证码、银行卡号、密码或身份证信息。建议挂断后联系已绑定的家属或社区人员。"
        helper_advice = "建议协助老人通过官方电话、银行官方渠道或独立的可信联系方式核实，不要直接回拨可疑号码。"
        warning = "这通电话风险较高，先停止操作，再找可信的人确认。"
    elif level == "MEDIUM":
        why = "该通话存在一定风险：声音证据不确定，或文本内容出现需要进一步核实的可疑信号。"
        elder_advice = "不要马上相信对方，不要立刻转账或提供个人信息。先让家属或社区人员帮忙确认。"
        helper_advice = "建议查看转录内容，确认对方身份、来电目的和是否涉及资金或验证码。"
        warning = "先核实身份，再决定是否继续沟通。"
    elif level == "VERIFY":
        why = "当前没有发现明显高危话术，且声音分数没有显示强伪造信号；但该证据来源需要结合身份核实，不能仅凭一次声学分数判断安全。"
        elder_advice = "先不要转账或提供验证码，请等待家属/社区人员确认对方身份。"
        helper_advice = "建议结合来电号码、真实亲属联系方式和更清晰的原始语音进一步核实。"
        warning = "当前更适合人工核实，不建议直接放行。"
    else:
        why = "当前音频的 AI 伪造概率较低，文本中没有发现明显的转账、验证码或紧急施压信号。"
        elder_advice = "当前风险较低，但只要对方后续提到钱、验证码、银行卡或密码，仍需立刻停止并核实。"
        helper_advice = "建议保留记录并提醒老人，涉及资金或验证码时必须二次确认。"
        warning = "即使当前风险较低，涉及钱和验证码也要先确认。"

    return f"""
1. 风险等级：{level_label(level, "zh")}
2. 为什么有风险：{why}
3. 声音证据：CNN 输出 AI 伪造语音概率 {fake:.2f}，真实语音概率 {real:.2f}。该分数仅作为辅助声学证据。证据来源为「{source_label}」，证据置信度为「{confidence_label}」。{evidence_note}
4. 文本证据：风险类型为「{scam_label(scam, "zh")}」；风险信号为「{format_signals(signals, "zh")}」。原始转录摘要：{text[:160]}
5. 给老人的建议：{elder_advice}
6. 给家属/社区的处理建议：{helper_advice}
7. 一句话提醒：{warning}
""".strip()


def analyze_with_gemma(
    quick: Dict[str, Any],
    elder_name: str = "老人",
    lang: str = "zh",
) -> Dict[str, Any]:
    lang = normalize_lang(lang)
    client = get_gemma_client()
    if client is None:
        report = local_safety_reasoning(quick, elder_name, lang)
        return {"source": "local_reasoning_no_hf_token", "report": report, "lang": lang}
    try:
        prompt = build_gemma_prompt(quick, elder_name, lang)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=520,
            temperature=0.2,
        )
        report = response.choices[0].message.content.strip()
        if not report or len(report) < 20:
            raise ValueError("Gemma output too short")
        return {"source": "gemma4_api", "report": report, "lang": lang}
    except Exception as e:
        print("Gemma API failed:", repr(e))
        report = local_safety_reasoning(quick, elder_name, lang)
        return {
            "source": "local_reasoning_gemma_error",
            "report": report,
            "error": repr(e),
            "lang": lang,
        }


def elder_simple_notice(level: str, lang: str = "zh") -> str:
    lang = normalize_lang(lang)
    if lang == "en":
        if level == "HIGH":
            return "High risk: Do not transfer money or provide verification codes, bank details, or passwords. Contact a trusted family or community helper immediately."
        if level == "MEDIUM":
            return "Caution: Verify the caller's identity first. Do not immediately transfer money or provide personal information. Ask a trusted helper to assist."
        if level == "VERIFY":
            return "Needs verification: No obvious high-risk signal was found, but the evidence source still requires human verification. Do not transfer money or provide verification codes before confirmation."
        return "Relatively low risk: No obvious high-risk signal was found, but always verify requests involving money, verification codes, bank information, or passwords."

    if level == "HIGH":
        return "高风险：不要转账，不要提供验证码、银行卡号或密码。请立即联系已绑定的家属或社区人员。"
    if level == "MEDIUM":
        return "谨慎：请先核实对方身份，不要立刻转账或提供个人信息。建议让家属/社区人员协助确认。"
    if level == "VERIFY":
        return "需核实：当前没有明显高危信号，但证据来源需要家属/社区人员确认。不要在确认前转账或提供验证码。"
    return "相对安全：当前未发现明显高危信号，但涉及钱、验证码、银行卡或密码时仍要先确认。"

# =========================================================
# HTML helpers
# =========================================================

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: radial-gradient(circle at top left, #1e3a8a 0, #08111f 34%, #020617 100%);
  color: #f8fafc;
  font-family: Arial, 'Microsoft YaHei', sans-serif;
}
a { color: inherit; }
.container { width: min(1180px, 94vw); margin: 0 auto; padding: 28px 0 48px; }
.nav { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 20px; }
.brand { font-size: 28px; font-weight: 900; }
.navlinks { display: flex; gap: 10px; flex-wrap: wrap; }
.navlinks a { text-decoration: none; background: rgba(37,99,235,.25); border: 1px solid rgba(147,197,253,.35); padding: 10px 14px; border-radius: 14px; font-weight: 800; }
.hero { background: linear-gradient(135deg, rgba(30,58,138,.95), rgba(15,23,42,.95)); border: 1px solid rgba(147,197,253,.55); border-radius: 28px; padding: 30px; box-shadow: 0 20px 70px rgba(0,0,0,.35); }
.hero h1 { font-size: 44px; margin: 0 0 12px; }
.hero p { color: #dbeafe; font-size: 19px; line-height: 1.65; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.card { background: rgba(15,23,42,.94); border: 1px solid rgba(96,165,250,.42); border-radius: 22px; padding: 22px; box-shadow: 0 14px 40px rgba(0,0,0,.26); }
.card h2 { margin-top: 0; font-size: 25px; color: #dbeafe; }
.card h3 { color: #bfdbfe; }
.muted { color: #cbd5e1; line-height: 1.65; }
.warning { background: rgba(248,113,113,.16); border: 1px solid rgba(248,113,113,.45); border-radius: 16px; padding: 14px; font-size: 18px; font-weight: 850; }
.good { background: rgba(34,197,94,.15); border: 1px solid rgba(74,222,128,.38); border-radius: 16px; padding: 14px; }
.input, select { width: 100%; margin: 8px 0 12px; padding: 14px; border-radius: 14px; background: #0f172a; border: 1px solid #60a5fa; color: white; font-size: 16px; }
button { width: 100%; margin: 8px 0; padding: 15px 18px; border: 1px solid #93c5fd; border-radius: 16px; background: #2563eb; color: white; font-size: 17px; font-weight: 900; cursor: pointer; }
button:hover { background: #1d4ed8; }
button.secondary { background: rgba(15,23,42,.65); }
button.danger { background: #be123c; border-color: #fecdd3; }
button.goodbtn { background: #15803d; border-color: #bbf7d0; }
.row { display: flex; gap: 10px; align-items: center; }
.row > * { flex: 1; }
.badge { display: inline-block; padding: 5px 9px; border-radius: 999px; background: rgba(37,99,235,.25); border: 1px solid rgba(147,197,253,.35); color: #dbeafe; font-weight: 800; margin: 3px; }
.list-item { border: 1px solid rgba(148,163,184,.35); background: rgba(2,6,23,.35); border-radius: 16px; padding: 14px; margin: 10px 0; }
pre { white-space: pre-wrap; word-break: break-word; background: rgba(2,6,23,.55); border: 1px solid rgba(148,163,184,.35); border-radius: 16px; padding: 16px; line-height: 1.55; }
.small { font-size: 13px; color: #94a3b8; }
.hidden { display: none !important; }
.audio-box { border: 1px dashed rgba(147,197,253,.55); border-radius: 16px; padding: 14px; margin-top: 12px; }
.elder-card { grid-column: 1 / -1; }
.elder-home { max-width: 980px; margin: 0 auto; }
.elder-greeting { font-size: 34px; font-weight: 950; margin: 0 0 18px; }
.guardian-status { border-radius: 22px; padding: 22px; margin: 14px 0 20px; }
.guardian-status.ready { background: rgba(34,197,94,.16); border: 1px solid rgba(74,222,128,.48); }
.guardian-status.waiting { background: rgba(245,158,11,.14); border: 1px solid rgba(251,191,36,.48); }
.guardian-status h2 { margin: 0 0 8px; font-size: 28px; }
.guardian-status p { margin: 6px 0; font-size: 19px; line-height: 1.55; }
.elder-sos { min-height: 112px; font-size: 30px; line-height: 1.25; border-radius: 24px; box-shadow: 0 16px 38px rgba(190,18,60,.28); }
.elder-sos span { display: block; margin-top: 7px; font-size: 18px; font-weight: 750; }
.elder-sos:disabled { cursor: not-allowed; opacity: .55; background: #64748b; border-color: #cbd5e1; }
.sync-note { text-align: center; color: #bfdbfe; font-size: 14px; margin: 9px 0 4px; }
.pending-card { background: rgba(245,158,11,.16); border: 2px solid rgba(251,191,36,.7); border-radius: 20px; padding: 18px; margin: 18px 0; }
.pending-card h3 { margin-top: 0; font-size: 24px; color: #fef3c7; }
.pending-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.code-box { display: block; width: 100%; margin: 12px 0; padding: 18px; text-align: center; border-radius: 18px; background: rgba(37,99,235,.25); border: 2px solid rgba(147,197,253,.65); font-size: 34px; font-weight: 950; letter-spacing: .22em; color: #eff6ff; }
.settings-panel { margin-top: 20px; border: 1px solid rgba(148,163,184,.35); border-radius: 18px; background: rgba(2,6,23,.28); overflow: hidden; }
.settings-panel summary { cursor: pointer; padding: 18px; font-size: 18px; font-weight: 850; color: #dbeafe; }
.settings-body { padding: 0 18px 18px; }
.demo-panel { margin-bottom: 18px; }
.demo-panel summary { cursor: pointer; color: #bfdbfe; font-weight: 850; padding: 10px 0; }
.form-label { display: block; margin-top: 12px; font-size: 17px; font-weight: 850; color: #dbeafe; }
.simple-record-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.elder-records summary { cursor: pointer; font-size: 20px; font-weight: 900; color: #dbeafe; }
@media (max-width: 860px) { .grid, .grid3, .pending-actions, .simple-record-grid { grid-template-columns: 1fr; } .hero h1 { font-size: 34px; } .nav, .row { flex-direction: column; align-items: stretch; } .elder-sos { font-size: 25px; min-height: 104px; } .code-box { font-size: 29px; letter-spacing: .16em; } }
body.i18n-loading { visibility: hidden; }
.nav-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }
.lang-switch { display: flex; align-items: center; gap: 6px; padding: 4px; border: 1px solid rgba(147,197,253,.35); border-radius: 14px; background: rgba(15,23,42,.62); }
.lang-switch button { width: auto; margin: 0; padding: 8px 11px; border-radius: 10px; background: transparent; border: 0; color: #bfdbfe; font-size: 14px; }
.lang-switch button:hover, .lang-switch button.active { background: rgba(37,99,235,.7); color: white; }
@media (max-width: 860px) { .nav-right { width: 100%; align-items: stretch; } .lang-switch { align-self: flex-start; } }
"""



# Chinese -> English UI dictionary.
# The app keeps one set of HTML/JavaScript templates and translates user-facing
# DOM text at runtime, avoiding duplicated elder/helper pages.
EN_REPLACEMENTS = {'首页': 'Home',
 '老人端': 'Older Adult',
 '家属/社区端': 'Family / Community',
 '面向老年人的 AI 语音诈骗协助保护系统：老人端一键求助，系统自动通知已绑定的 3–5 位可信协助人；家属/社区端分两阶段完成 CNN + Whisper 证据分析与 Gemma 4 风险推理。': 'An AI voice-scam assistance system '
                                                                                                      'for older adults: the older-adult '
                                                                                                      'side can request help with one tap, '
                                                                                                      'the system notifies 3–5 trusted '
                                                                                                      'helpers, and the family/community '
                                                                                                      'side performs two-stage CNN + '
                                                                                                      'Whisper evidence analysis and Gemma '
                                                                                                      '4 risk reasoning.',
 '👵 老人端': '👵 Older Adult',
 '首次由家人协助完成手机号设置，以后无需重复登录。系统自动同步守护状态，老人只需一键求助。': 'A family member can help complete the phone setup once. After that, the device remembers '
                                                 'the session and the older adult can request help with one tap.',
 '进入老人端': 'Open Older-Adult Side',
 '👨\u200d👩\u200d👧 家属/社区端': '👨\u200d👩\u200d👧 Family / Community',
 '手机号登录，输入老人端显示的 6 位连接数字，等待老人确认后处理求助。': 'Sign in with a phone number, enter the six-digit code shown on the older-adult side, and wait for '
                                        'approval before handling assistance requests.',
 '进入家属/社区端': 'Open Family / Community Side',
 '👵 老人端：遇到可疑电话，一键找家人': '👵 Older Adult: Get Help with a Suspicious Call',
 '不需要理解 AI 分数，也不需要每天登录。第一次由家人帮助完成设置，以后打开页面即可直接求助。': 'You do not need to understand AI scores or sign in every day. A family member can '
                                                    'help with the first setup, and later you can open the page and request help directly.',
 '首次设置': 'First-Time Setup',
 '这不是身份证实名认证。只用手机号确认这是您的设备，通常由家人帮助完成一次即可。': 'This is not identity-card verification. The phone number is only used to confirm this device, '
                                            'usually with one-time help from a family member.',
 '比赛演示：快速进入老人账号': 'Demo: Quickly Open an Older-Adult Account',
 '仅用于现场 Demo，不需要手动输入验证码。': 'For demonstration only; no manual verification-code entry is required.',
 '怎么称呼您': 'How should we address you?',
 '例如：刘奶奶（不需要填写身份证姓名）': 'Example: Mrs. Liu (legal name is not required)',
 '您的手机号': 'Your Phone Number',
 '例如：18800000001': 'Example: 18800000001',
 '获取短信验证码': 'Get Verification Code',
 '短信中的数字': 'Verification Code',
 '请输入验证码': 'Enter the verification code',
 '完成设置，进入老人端': 'Complete Setup and Continue',
 '完成后设备会记住登录状态，下次无需重复操作。': 'The device will remember the session after setup, so you will not need to repeat this next time.',
 '🚨 我遇到了可疑电话': '🚨 I Received a Suspicious Call',
 '立即通知已经连接的家人': 'Notify My Trusted Helpers',
 '守护状态会自动更新，不需要手动刷新': 'Protection status updates automatically; no manual refresh is needed.',
 '需要保留声音证据吗？': 'Would you like to save audio evidence?',
 '先点击上面的红色求助按钮，再任选一种方式。不会录音也没有关系，家人仍会收到求助。': 'First tap the red help button above, then choose either option. If you cannot record audio, '
                                             'your trusted helpers will still receive the request.',
 '🎙️ 录制 12 秒': '🎙️ Record 12 Seconds',
 '📁 上传已有语音': '📁 Upload Existing Audio',
 '上传微信语音、留言或通话录音': 'Upload a Social Voice Message, Voicemail, or Call Recording',
 '微信/社交软件语音': 'Social-Media Voice Message',
 '保存的通话录音': 'Saved Call Recording',
 '语音留言': 'Voicemail',
 '其他已有音频文件': 'Other Existing Audio File',
 '上传这段语音': 'Upload This Audio',
 '家属设置（通常由家人操作）': 'Trusted-Helper Settings (Usually Managed by Family)',
 '更换老人账号': 'Switch Older-Adult Account',
 '查看我的求助记录': 'View My Assistance History',
 '手机号：': 'Phone: ',
 '解除与此人的连接': 'Remove This Trusted Connection',
 '有人想成为您的守护人': 'Someone Wants to Become Your Trusted Helper',
 '只有认识并信任这个人时，才点击同意。陌生人请点击“我不认识”。': "Approve only if you know and trust this person. If the requester is unfamiliar, choose “I Don't Know "
                                    'This Person.”',
 '是我的家人，同意': 'I Know and Trust This Person',
 '我不认识': "I Don't Know This Person",
 '请再次确认：您认识并信任这个人吗？': 'Please confirm: do you know and trust this person?',
 '确认拒绝这个陌生申请吗？': 'Reject this unfamiliar request?',
 '暂时无法更新守护状态': 'Unable to update protection status right now',
 '🚨 我遇到了可疑电话<span>请先让家人完成连接</span>': '🚨 I Received a Suspicious Call<span>Please connect a trusted helper first</span>',
 '✅ 家人守护已开启': '✅ Trusted Protection Is Active',
 '会收到您的求助。': ' will receive your assistance request.',
 '当前已连接 ': 'Currently connected to ',
 ' 位可信协助人。': ' trusted helper(s).',
 '还没有连接家人': 'No Trusted Helper Connected Yet',
 '请让家人在“家属/社区端”输入下面的 6 位数字。': 'Ask a trusted family or community helper to enter the six-digit code below on the Family / Community side.',
 '这串数字只告诉自己的家人或可信社区工作人员，不要告诉陌生人。': 'Share this code only with your own family or a trusted community worker. Do not share it with '
                                   'strangers.',
 '，您好': ', hello',
 '给家人的 6 位连接数字': 'Six-Digit Trusted-Helper Code',
 '家人输入后，申请会自动出现在主页，不需要点击“刷新”。': 'After a trusted helper enters the code, the request will appear automatically. No manual refresh is '
                                'needed.',
 '已经连接的人': 'Connected Trusted Helpers',
 '暂时没有连接任何人。': 'No trusted helper is connected yet.',
 '守护状态已自动更新 · 不需要手动刷新': 'Protection status updated automatically · no manual refresh needed',
 '网络暂时不稳定，系统稍后会自动重试': 'The network is temporarily unstable. The system will retry automatically.',
 '确认解除与这个人的连接吗？解除后，对方将收不到新的求助。': 'Remove this trusted connection? This person will no longer receive new assistance requests.',
 '已解除连接。': 'Connection removed.',
 '还没有连接家人，请先让家人完成设置。': 'No trusted helper is connected yet. Please complete helper setup first.',
 '正在通知家人，请稍等……': 'Notifying your trusted helpers. Please wait...',
 '求助已经发出，已通知 ': 'Assistance request sent. Notified ',
 ' 位家人。': ' trusted helper(s).',
 '现在请不要转账，不要提供验证码、银行卡号或密码。需要时可在下方保留声音证据。': 'Do not transfer money or provide verification codes, bank details, or passwords. You can save '
                                           'audio evidence below if needed.',
 '求助发送失败：': 'Failed to send assistance request: ',
 '已通知：': 'Notified: ',
 '暂无': 'None',
 '已保留声音证据。': 'Audio evidence saved.',
 '证据来源：': 'Evidence Source: ',
 '证据置信度：': 'Evidence Confidence: ',
 '正在等待家人处理。': 'Waiting for a trusted helper to review the request.',
 '暂无求助记录。': 'No assistance history yet.',
 '请先点击上面的红色求助按钮，再开始录音。': 'Tap the red help button above before starting a recording.',
 '正在上传录音……': 'Uploading recording...',
 '录音已经保存，家人可以查看并处理。': 'Recording saved. Your trusted helper can now review it.',
 '正在录制 12 秒。不要说出验证码、密码或银行卡号。': 'Recording for 12 seconds. Do not say verification codes, passwords, or bank details aloud.',
 '录音失败：': 'Recording failed: ',
 '请先点击上面的红色求助按钮，再上传语音。': 'Tap the red help button above before uploading audio.',
 '请先选择一段语音。': 'Please select an audio file first.',
 '正在上传语音……': 'Uploading audio...',
 '语音已经保存，家人可以查看并处理。': 'Audio saved. Your trusted helper can now review it.',
 '上传失败：': 'Upload failed: ',
 '👨\u200d👩\u200d👧 家属/社区端：可信协助人网络': '👨\u200d👩\u200d👧 Family / Community: Trusted Helper Network',
 '家属或社区工作人员输入老人端显示的 6 位连接数字并提交申请。老人确认后，求助会进入你的收件箱。': 'A family member or community worker enters the six-digit code shown on the '
                                                     'older-adult side and submits a request. After approval, assistance requests will '
                                                     'appear in your inbox.',
 '手机号登录 / 注册协助人账号': 'Sign In / Register a Helper Account',
 '姓名/身份，例如：女儿': 'Name / role, for example: Daughter',
 '家属 family': 'Family',
 '社区工作人员 community': 'Community Worker',
 '志愿者 volunteer': 'Volunteer',
 '手机号，例如：18800000002': 'Phone number, e.g. 18800000002',
 '发送短信验证码': 'Send Verification Code',
 '输入验证码': 'Enter Verification Code',
 '登录协助人端': 'Sign In to Helper Side',
 'HF Demo 中验证码会显示在页面上；真实上线时接入短信服务商。': 'In the HF demo, the verification code is displayed on the page. A production deployment would use '
                                      'an SMS provider.',
 '绑定老人': 'Connect an Older Adult',
 '输入老人端的 6 位数字，例如 438216': "Enter the older adult's six-digit code, e.g. 438216",
 '发送连接申请': 'Send Connection Request',
 '刷新': 'Refresh',
 '我的待处理请求': 'My Pending Assistance Requests',
 '两阶段风险分析': 'Two-Stage Risk Analysis',
 '请选择一个请求。': 'Please select a request.',
 '第一阶段：CNN + Whisper': 'Stage 1: CNN + Whisper',
 '可以使用老人端保留证据，也可以由家属/社区端上传通话录音、语音留言或社交软件语音。系统会记录证据来源，并在风险结果中给出相应的核实提醒。': 'Use audio saved by the older adult or upload a call recording, '
                                                                         'voicemail, or social-media voice message. The system records the '
                                                                         'evidence source and provides source-aware verification guidance.',
 '家属/社区端上传音频': 'Helper-Uploaded Audio',
 '上传音频并执行 CNN + Whisper': 'Upload Audio and Run CNN + Whisper',
 '使用老人端保留证据执行 CNN + Whisper': 'Run CNN + Whisper on Older-Adult Evidence',
 '第二阶段：Gemma 4 API 风险推理': 'Stage 2: Gemma 4 Risk Reasoning',
 'Gemma 4 综合 CNN 声学证据、Whisper 文本证据和诈骗行为信号生成老人友好的建议。': 'Gemma 4 combines CNN acoustic evidence, Whisper transcript evidence, and '
                                                      'scam-behavior signals to generate understandable safety guidance.',
 '执行 Gemma 4 风险推理': 'Run Gemma 4 Risk Reasoning',
 '解绑此老人': 'Disconnect This Older Adult',
 '等待老人确认': 'Waiting for Older-Adult Approval',
 '申请时间：': 'Requested at: ',
 '绑定申请已发送到老人端。老人确认后，你才会正式成为协助人。': 'The connection request was sent to the older-adult side. You will become a trusted helper only after '
                                  'approval.',
 '登录失效': 'Session expired',
 '当前协助人：': 'Current Helper: ',
 '已绑定老人：': 'Connected Older Adults: ',
 '还没有正式绑定老人。请输入老人端显示的家庭守护绑定码，并等待老人端确认。': 'No older adult is connected yet. Enter the six-digit trusted-helper code and wait for approval.',
 '暂无待确认绑定申请。': 'No pending connection requests.',
 '确认解除与这个老人的绑定吗？': 'Disconnect from this older adult?',
 '发起求助': ' requested help',
 '有老人端证据': 'Older-Adult Evidence Available',
 '等待音频': 'Waiting for Audio',
 '选择处理': 'Select',
 '暂无待处理请求。': 'No pending assistance requests.',
 '当前处理请求：': 'Current Request: ',
 'CNN 声学辅助证据：': 'CNN Acoustic Evidence: ',
 'AI伪造概率 ': 'AI-spoof probability ',
 '，真实语音概率 ': ', real-speech probability ',
 '未知': 'Unknown',
 '中等': 'Medium',
 '证据使用提醒：': 'Evidence Note: ',
 '请结合家属/社区人员核实。': 'Please verify with a trusted family or community helper.',
 'Whisper 转录：': 'Whisper Transcript: ',
 '风险类型：': 'Risk Type: ',
 '风险信号：': 'Risk Signals: ',
 '处理时间：': 'Processing Time: ',
 ' 秒': ' s',
 'Gemma 4 风险推理': 'Gemma 4 Risk Reasoning',
 '来源：': 'Source: ',
 '请先选择一个请求。': 'Please select a request first.',
 '请先选择音频文件。': 'Please select an audio file first.',
 '正在执行 CNN + Whisper，请稍等...': 'Running CNN + Whisper. Please wait...',
 '正在使用老人端证据执行 CNN + Whisper，请稍等...': 'Running CNN + Whisper on older-adult evidence. Please wait...',
 '正在调用 Gemma 4 API 生成风险推理...': 'Generating Gemma 4 risk reasoning...',
 '比赛 Demo 验证码：': 'Demo verification code: ',
 'Demo 验证码：': 'Demo verification code: ',
 '（真实上线时通过短信发送）': ' (sent by SMS in production)',
 '验证码已生成。HF Demo 中直接显示验证码；真实上线时由短信服务商发送。': 'Verification code generated. It is displayed directly in the HF demo; a production deployment '
                                           'would send it through an SMS provider.',
 '请输入有效手机号': 'Please enter a valid phone number',
 '请先发送验证码': 'Please request a verification code first',
 '验证码已过期，请重新发送': 'The verification code has expired. Please request a new one.',
 '验证码错误': 'Incorrect verification code',
 '该手机号已注册为另一种角色。Demo 中请使用另一个手机号。': 'This phone number is already registered with another role. Please use another number in the demo.',
 '请输入老人端显示的 6 位连接数字': 'Enter the six-digit connection code shown on the older-adult side',
 '连接数字应为 6 位': 'The connection code must contain six digits',
 '协助人账号不存在，请重新登录': 'Helper account not found. Please sign in again.',
 '没有找到该绑定码对应的老人账号': 'No older-adult account was found for this connection code',
 '绑定申请已发送给 ': 'Connection request sent to ',
 '，请等待老人端确认。': '. Please wait for older-adult approval.',
 '绑定申请不存在': 'Connection request not found',
 '你无权处理这个绑定申请': 'You do not have permission to process this connection request',
 '申请人账号不存在，已自动拒绝': 'Requester account not found; the request was automatically rejected',
 '已拒绝该绑定申请。': 'Connection request rejected.',
 '成为你的可信协助人。': ' is now your trusted helper.',
 '缺少 helper_id': 'Missing helper_id',
 '已解绑该协助人': 'Trusted helper disconnected',
 '缺少 elder_id': 'Missing elder_id',
 '已解除与该老人的绑定': 'Disconnected from the older adult',
 '还没有绑定协助人，请先绑定。': 'No trusted helper is connected yet. Please connect one first.',
 '请求不存在': 'Request not found',
 '你无权查看该请求': 'You do not have permission to view this request',
 '你无权上传该请求证据': 'You do not have permission to upload evidence for this request',
 '你无权处理该请求': 'You do not have permission to process this request',
 '没有可用音频。请上传录音，或先让老人端保留证据。': 'No audio is available. Upload a recording or ask the older-adult side to save audio evidence first.',
 'CNN + Whisper 扫描失败：': 'CNN + Whisper scan failed: ',
 '请先完成第一阶段 CNN + Whisper 扫描': 'Complete Stage 1 CNN + Whisper analysis first',
 '高风险': 'High Risk',
 '谨慎': 'Caution',
 '需核实': 'Needs Verification',
 '相对安全': 'Relatively Safe',
 'AI 语音伪造风险': 'AI voice spoofing risk',
 '金融或投资诈骗风险': 'financial or investment scam',
 '冒充亲属诈骗风险': 'family impersonation scam',
 '验证码诈骗风险': 'verification-code scam',
 '未知或不明确风险': 'unknown or unclear risk',
 '转账/资金相关信号': 'money-transfer / financial signal',
 '验证码/密码相关信号': 'verification-code / password signal',
 '紧急施压信号': 'urgency-pressure signal',
 '冒充亲属信号': 'family-impersonation signal',
 '投资收益诱导信号': 'investment-profit signal',
 '未发现明显关键词风险信号': 'no obvious keyword risk signal',
 '需结合人工核实': 'Human verification required',
 '较高': 'Relatively high',
 '老人端现场录音': 'Older-adult microphone recording',
 '老人端上传音频': 'Older-adult uploaded audio',
 '该证据来自浏览器/麦克风现场保留，适合快速求助，但可能受到播放设备、距离和环境噪声影响，不建议仅凭声学分数判断。': 'This evidence was captured through a browser or microphone. Playback '
                                                             'devices, distance, and background noise may reduce acoustic reliability; do '
                                                             'not rely on the acoustic score alone.',
 '该证据由老人端上传，通常比现场录音更稳定，但仍建议结合通话内容和家属核实。': 'This evidence was uploaded by the older adult and is usually more stable than a live microphone '
                                          'recording, but it should still be combined with conversation content and trusted-human '
                                          'verification.',
 '该证据可能经过社交软件压缩，适合判断话术风险，但声学判断仍需谨慎。': 'This evidence may have been compressed by a social-media platform. It is useful for semantic risk '
                                      'analysis, but acoustic conclusions should be treated cautiously.',
 '该证据来自语音留言，适合转录和风险话术分析。': 'This evidence comes from voicemail and is suitable for transcription and scam-language analysis.',
 '该证据来自已保存通话录音，通常比现场麦克风重录更适合声学分析。': 'This evidence comes from a saved call recording and is generally more suitable for acoustic analysis '
                                    'than a live microphone re-recording.',
 '该证据由家属/社区端补充上传，适合进行 CNN + Whisper 分析。': 'This evidence was uploaded by a trusted helper and is suitable for CNN + Whisper analysis.'}

LANGUAGE_JS = r"""
<script>
(function() {
  const EN_MAP = __EN_MAP__;
  const SORTED_EN_KEYS = Object.keys(EN_MAP).sort((a, b) => b.length - a.length);

  function normalizeLang(value) {
    value = String(value || '').toLowerCase();
    return value.startsWith('en') ? 'en' : 'zh';
  }

  const params = new URLSearchParams(window.location.search);
  const queryLang = params.get('lang');
  const savedLang = localStorage.getItem('gemmashield_lang');
  const initialLang = normalizeLang(queryLang || savedLang || 'zh');

  window.GEMMASHIELD_LANG = initialLang;
  localStorage.setItem('gemmashield_lang', initialLang);
  document.documentElement.lang = initialLang === 'en' ? 'en' : 'zh-CN';

  window.setAppLanguage = function(lang) {
    lang = normalizeLang(lang);
    localStorage.setItem('gemmashield_lang', lang);
    const url = new URL(window.location.href);
    url.searchParams.set('lang', lang);
    window.location.href = url.toString();
  };

  window.gsTranslate = function(value) {
    if (window.GEMMASHIELD_LANG !== 'en' || value === null || value === undefined) return value;
    let out = String(value);
    for (const key of SORTED_EN_KEYS) {
      if (out.includes(key)) out = out.split(key).join(EN_MAP[key]);
    }
    return out;
  };

  window.gsQ = function(q, base) {
    if (!q) return '';
    const lang = window.GEMMASHIELD_LANG === 'en' ? 'en' : 'zh';
    return q[`${base}_${lang}`] ?? q[`${base}_zh`] ?? q[base] ?? '';
  };

  window.gsGemmaResult = function(requestObj) {
    if (!requestObj) return null;
    if (window.GEMMASHIELD_LANG === 'en') {
      return requestObj.gemma_result_en || null;
    }
    return requestObj.gemma_result_zh || requestObj.gemma_result || null;
  };

  window.gsElderSimpleResult = function(requestObj) {
    if (!requestObj) return '';
    if (window.GEMMASHIELD_LANG === 'en') {
      return requestObj.elder_simple_result_en || window.gsTranslate(requestObj.elder_simple_result || '');
    }
    return requestObj.elder_simple_result_zh || requestObj.elder_simple_result || '';
  };

  function shouldSkip(node) {
    const parent = node && (node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement);
    return !!(parent && parent.closest && parent.closest('[data-no-i18n="true"]'));
  }

  function translateElement(el) {
    if (!el || shouldSkip(el)) return;
    if (el.nodeType === Node.TEXT_NODE) {
      const original = el.nodeValue;
      const translated = window.gsTranslate(original);
      if (translated !== original) el.nodeValue = translated;
      return;
    }
    if (el.nodeType !== Node.ELEMENT_NODE) return;

    for (const attr of ['placeholder', 'title', 'aria-label']) {
      if (el.hasAttribute && el.hasAttribute(attr)) {
        const original = el.getAttribute(attr);
        const translated = window.gsTranslate(original);
        if (translated !== original) el.setAttribute(attr, translated);
      }
    }

    for (const child of Array.from(el.childNodes || [])) translateElement(child);
  }

  function translateDocument() {
    if (window.GEMMASHIELD_LANG === 'en') {
      translateElement(document.body);
      const helperName = document.getElementById('helperName');
      if (helperName && helperName.value === '女儿') helperName.value = 'Daughter';
    }
    document.body.classList.remove('i18n-loading');

    // Keep the selected language when moving between app pages.
    for (const a of document.querySelectorAll('a[href^="/"]')) {
      const href = a.getAttribute('href');
      if (!href || href.startsWith('/api/')) continue;
      try {
        const u = new URL(href, window.location.origin);
        u.searchParams.set('lang', window.GEMMASHIELD_LANG);
        a.setAttribute('href', u.pathname + u.search + u.hash);
      } catch (_) {}
    }

    document.querySelectorAll('[data-lang-button]').forEach(btn => {
      const active = btn.getAttribute('data-lang-button') === window.GEMMASHIELD_LANG;
      btn.classList.toggle('active', active);
    });
  }

  // Translate runtime alerts and confirmation dialogs.
  const nativeAlert = window.alert.bind(window);
  window.alert = function(message) {
    return nativeAlert(window.gsTranslate(message));
  };

  const nativeConfirm = window.confirm.bind(window);
  window.confirm = function(message) {
    return nativeConfirm(window.gsTranslate(message));
  };

  // Automatically attach the current language to API requests.
  const nativeFetch = window.fetch.bind(window);
  window.fetch = function(input, init) {
    init = init ? {...init} : {};
    let requestInput = input;

    if (typeof requestInput === 'string' && requestInput.startsWith('/api/')) {
      const u = new URL(requestInput, window.location.origin);
      if (!u.searchParams.has('lang')) u.searchParams.set('lang', window.GEMMASHIELD_LANG);
      requestInput = u.pathname + u.search + u.hash;
    }

    if (init.body instanceof FormData && !init.body.has('lang')) {
      init.body.append('lang', window.GEMMASHIELD_LANG);
    }

    const headers = new Headers(init.headers || {});
    const contentType = headers.get('Content-Type') || headers.get('content-type') || '';
    if (typeof init.body === 'string' && contentType.includes('application/json')) {
      try {
        const payload = JSON.parse(init.body);
        if (payload && typeof payload === 'object' && !Array.isArray(payload) && !payload.lang) {
          payload.lang = window.GEMMASHIELD_LANG;
          init.body = JSON.stringify(payload);
        }
      } catch (_) {}
    }

    return nativeFetch(requestInput, init);
  };

  document.addEventListener('DOMContentLoaded', () => {
    translateDocument();

    if (window.GEMMASHIELD_LANG === 'en') {
      const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          if (mutation.type === 'characterData') translateElement(mutation.target);
          for (const node of Array.from(mutation.addedNodes || [])) translateElement(node);
        }
      });
      observer.observe(document.body, {subtree: true, childList: true, characterData: true});
    }
  });
})();
</script>
""".replace("__EN_MAP__", json.dumps(EN_REPLACEMENTS, ensure_ascii=False))


def page_shell(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>{CSS}</style>
  {LANGUAGE_JS}
</head>
<body class="i18n-loading">
  <div class="container">
    <div class="nav">
      <div class="brand">🛡️ GemmaShield</div>
      <div class="nav-right">
        <div class="navlinks">
          <a href="/">首页</a>
          <a href="/elder">老人端</a>
          <a href="/helper">家属/社区端</a>
          <a href="/health">Health</a>
        </div>
        <div class="lang-switch" aria-label="Language">
          <button type="button" data-lang-button="zh" onclick="setAppLanguage('zh')">中文</button>
          <button type="button" data-lang-button="en" onclick="setAppLanguage('en')">English</button>
        </div>
      </div>
    </div>
    {body}
  </div>
</body>
</html>"""


ROOT_HTML = page_shell("GemmaShield", """
<div class="hero">
  <h1>GemmaShield Trusted Helper Network</h1>
  <p>面向老年人的 AI 语音诈骗协助保护系统：老人端一键求助，系统自动通知已绑定的 3–5 位可信协助人；家属/社区端分两阶段完成 CNN + Whisper 证据分析与 Gemma 4 风险推理。</p>
  <div class="grid">
    <div class="card">
      <h2>👵 老人端</h2>
      <p class="muted">首次由家人协助完成手机号设置，以后无需重复登录。系统自动同步守护状态，老人只需一键求助。</p>
      <a href="/elder"><button>进入老人端</button></a>
    </div>
    <div class="card">
      <h2>👨‍👩‍👧 家属/社区端</h2>
      <p class="muted">手机号登录，输入老人端显示的 6 位连接数字，等待老人确认后处理求助。</p>
      <a href="/helper"><button>进入家属/社区端</button></a>
    </div>
  </div>
</div>
""")


ELDER_HTML = page_shell("GemmaShield Elder", r"""
<div class="hero">
  <h1>👵 老人端：遇到可疑电话，一键找家人</h1>
  <p>不需要理解 AI 分数，也不需要每天登录。第一次由家人帮助完成设置，以后打开页面即可直接求助。</p>
</div>

<div class="grid">
  <div class="card elder-card" id="loginCard">
    <h2>首次设置</h2>
    <p class="muted">这不是身份证实名认证。只用手机号确认这是您的设备，通常由家人帮助完成一次即可。</p>

    <details class="demo-panel">
      <summary>比赛演示：快速进入老人账号</summary>
      <p class="muted">仅用于现场 Demo，不需要手动输入验证码。</p>
      <div class="row">
        <button class="secondary" onclick="demoLoginElder('刘奶奶','18800000001')">刘奶奶</button>
        <button class="secondary" onclick="demoLoginElder('王爷爷','18800000003')">王爷爷</button>
        <button class="secondary" onclick="demoLoginElder('张阿姨','18800000005')">张阿姨</button>
      </div>
    </details>

    <label class="form-label" for="elderName">怎么称呼您</label>
    <input class="input" id="elderName" placeholder="例如：刘奶奶（不需要填写身份证姓名）" value="刘奶奶" />
    <label class="form-label" for="elderPhone">您的手机号</label>
    <input class="input" id="elderPhone" inputmode="tel" placeholder="例如：18800000001" value="18800000001" />
    <button onclick="sendOtp('elder')">获取短信验证码</button>
    <div id="otpNotice" class="small"></div>
    <label class="form-label" for="elderOtp">短信中的数字</label>
    <input class="input" id="elderOtp" inputmode="numeric" maxlength="6" placeholder="请输入验证码" />
    <button class="goodbtn" onclick="verifyOtp('elder')">完成设置，进入老人端</button>
    <p class="muted">完成后设备会记住登录状态，下次无需重复操作。</p>
  </div>

  <div class="card hidden elder-card elder-home" id="profileCard">
    <div id="elderProfile"></div>

    <button class="danger elder-sos" id="helpButton" onclick="createHelpRequest()">
      🚨 我遇到了可疑电话
      <span>立即通知已经连接的家人</span>
    </button>
    <div id="helpStatus"></div>
    <div id="syncStatus" class="sync-note">守护状态会自动更新，不需要手动刷新</div>

    <div class="audio-box">
      <h3>需要保留声音证据吗？</h3>
      <p class="muted">先点击上面的红色求助按钮，再任选一种方式。不会录音也没有关系，家人仍会收到求助。</p>
      <div class="simple-record-grid">
        <button class="secondary" onclick="recordEvidence()">🎙️ 录制 12 秒</button>
        <button class="secondary" onclick="document.getElementById('uploadPanel').open = true">📁 上传已有语音</button>
      </div>
      <details id="uploadPanel" class="settings-panel">
        <summary>上传微信语音、留言或通话录音</summary>
        <div class="settings-body">
          <select id="elderUploadSource" class="input">
            <option value="wechat_voice">微信/社交软件语音</option>
            <option value="call_recording">保存的通话录音</option>
            <option value="voicemail">语音留言</option>
            <option value="elder_upload">其他已有音频文件</option>
          </select>
          <input type="file" id="elderEvidenceFile" class="input" accept="audio/*" />
          <button class="secondary" onclick="uploadEvidenceFile()">上传这段语音</button>
        </div>
      </details>
      <div id="recordStatus" class="small"></div>
    </div>

    <details class="settings-panel">
      <summary>家属设置（通常由家人操作）</summary>
      <div class="settings-body" id="elderSettings"></div>
      <div class="settings-body">
        <button class="secondary" onclick="switchElderAccount()">更换老人账号</button>
      </div>
    </details>
  </div>
</div>

<div class="card hidden elder-card elder-home elder-records" id="requestsCard">
  <details>
    <summary>查看我的求助记录</summary>
    <div id="elderRequests"></div>
  </details>
</div>

<script>
let elderToken = localStorage.getItem('gemmashield_elder_token') || '';
let latestRequestId = localStorage.getItem('gemmashield_latest_request_id') || '';
let elderRefreshTimer = null;
let elderRefreshBusy = false;

async function apiJSON(url, data) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
  const json = await res.json();
  if (!res.ok) throw new Error(json.detail || JSON.stringify(json));
  return json;
}

function relationshipLabel(value) {
  const zh = {family:'家人', community:'社区工作人员', volunteer:'志愿者'};
  const en = {family:'Family', community:'Community Worker', volunteer:'Volunteer'};
  const table = window.GEMMASHIELD_LANG === 'en' ? en : zh;
  return table[value] || (window.GEMMASHIELD_LANG === 'en' ? 'Trusted Helper' : '可信协助人');
}

function showLogin() {
  document.getElementById('loginCard').classList.remove('hidden');
  document.getElementById('profileCard').classList.add('hidden');
  document.getElementById('requestsCard').classList.add('hidden');
}

function showProfile() {
  document.getElementById('loginCard').classList.add('hidden');
  document.getElementById('profileCard').classList.remove('hidden');
  document.getElementById('requestsCard').classList.remove('hidden');
}

function stopElderAutoRefresh() {
  if (elderRefreshTimer) clearInterval(elderRefreshTimer);
  elderRefreshTimer = null;
}

function startElderAutoRefresh() {
  stopElderAutoRefresh();
  refreshElder();
  elderRefreshTimer = setInterval(() => {
    if (elderToken && !document.hidden) refreshElder(true);
  }, 5000);
}

function switchElderAccount() {
  stopElderAutoRefresh();
  localStorage.removeItem('gemmashield_elder_token');
  localStorage.removeItem('gemmashield_latest_request_id');
  elderToken = '';
  latestRequestId = '';
  document.getElementById('otpNotice').innerHTML = '';
  document.getElementById('elderOtp').value = '';
  document.getElementById('elderRequests').innerHTML = '';
  document.getElementById('helpStatus').innerHTML = '';
  showLogin();
}

async function demoLoginElder(name, phone) {
  try {
    switchElderAccount();
    document.getElementById('elderName').value = name;
    document.getElementById('elderPhone').value = phone;
    const otp = await apiJSON('/api/send_otp', {phone, role: 'elder', display_name: name});
    document.getElementById('otpNotice').innerHTML = `比赛 Demo 验证码：<b>${otp.demo_otp}</b>`;
    document.getElementById('elderOtp').value = otp.demo_otp;
    const data = await apiJSON('/api/verify_otp', {phone, role: 'elder', display_name: name, code: otp.demo_otp});
    elderToken = data.token;
    localStorage.setItem('gemmashield_elder_token', elderToken);
    startElderAutoRefresh();
  } catch(e) { alert(e.message); }
}

async function sendOtp(role) {
  const phone = document.getElementById('elderPhone').value;
  const display_name = document.getElementById('elderName').value || '老人';
  try {
    const data = await apiJSON('/api/send_otp', {phone, role, display_name});
    document.getElementById('otpNotice').innerHTML = `Demo 验证码：<b>${data.demo_otp}</b>（真实上线时通过短信发送）`;
    document.getElementById('elderOtp').value = data.demo_otp;
  } catch(e) { alert(e.message); }
}

async function verifyOtp(role) {
  const phone = document.getElementById('elderPhone').value;
  const display_name = document.getElementById('elderName').value || '老人';
  const code = document.getElementById('elderOtp').value;
  try {
    const data = await apiJSON('/api/verify_otp', {phone, role, display_name, code});
    elderToken = data.token;
    localStorage.setItem('gemmashield_elder_token', elderToken);
    latestRequestId = '';
    localStorage.removeItem('gemmashield_latest_request_id');
    startElderAutoRefresh();
  } catch(e) { alert(e.message); }
}

function helperHTML(h) {
  return `<div class="list-item">
    <b>${h.display_name}</b> <span class="badge">${relationshipLabel(h.relationship)}</span>
    <p class="small">手机号：${h.phone}</p>
    <button class="secondary" onclick="unbindHelper('${h.user_id}')">解除与此人的连接</button>
  </div>`;
}

function pendingBindingHTML(b) {
  return `<div class="pending-card">
    <h3>有人想成为您的守护人</h3>
    <p><b>${b.helper_name}</b>（${relationshipLabel(b.relationship)}，手机 ${b.helper_phone}）</p>
    <p class="muted">只有认识并信任这个人时，才点击同意。陌生人请点击“我不认识”。</p>
    <div class="pending-actions">
      <button class="goodbtn" onclick="decideBinding('${b.binding_request_id}', 'approve')">是我的家人，同意</button>
      <button class="danger" onclick="decideBinding('${b.binding_request_id}', 'reject')">我不认识</button>
    </div>
  </div>`;
}

async function decideBinding(bindingRequestId, action) {
  const text = action === 'approve'
    ? '请再次确认：您认识并信任这个人吗？'
    : '确认拒绝这个陌生申请吗？';
  if (!confirm(text)) return;
  try {
    const data = await apiJSON('/api/confirm_binding', {
      token: elderToken,
      binding_request_id: bindingRequestId,
      action
    });
    document.getElementById('helpStatus').innerHTML = `<p class="good">${data.message}</p>`;
    await refreshElder();
  } catch(e) { alert(e.message); }
}

async function refreshElder(silent = false) {
  if (!elderToken) { showLogin(); return; }
  if (elderRefreshBusy) return;
  elderRefreshBusy = true;
  try {
    const res = await fetch('/api/me?token=' + encodeURIComponent(elderToken));
    const data = await res.json();
    if (res.status === 401 || res.status === 403) {
      switchElderAccount();
      return;
    }
    if (!res.ok) throw new Error(data.detail || '暂时无法更新守护状态');

    showProfile();
    const u = data.user;
    const helpers = u.bound_helpers || [];
    const pending = u.pending_binding_requests || [];
    const helperNames = helpers.map(h => h.display_name).join('、');
    const helpButton = document.getElementById('helpButton');

    helpButton.disabled = helpers.length === 0;
    helpButton.innerHTML = helpers.length
      ? `🚨 我遇到了可疑电话<span>立即通知 ${helperNames}</span>`
      : `🚨 我遇到了可疑电话<span>请先让家人完成连接</span>`;

    const statusHTML = helpers.length
      ? `<div class="guardian-status ready">
           <h2>✅ 家人守护已开启</h2>
           <p><b>${helperNames}</b> 会收到您的求助。</p>
           <p class="small">当前已连接 ${helpers.length} 位可信协助人。</p>
         </div>`
      : `<div class="guardian-status waiting">
           <h2>还没有连接家人</h2>
           <p>请让家人在“家属/社区端”输入下面的 6 位数字。</p>
           <span class="code-box">${u.bind_code}</span>
           <p class="small">这串数字只告诉自己的家人或可信社区工作人员，不要告诉陌生人。</p>
         </div>`;

    document.getElementById('elderProfile').innerHTML = `
      <div class="elder-greeting">${u.display_name}，您好</div>
      ${statusHTML}
      ${pending.length ? pending.map(pendingBindingHTML).join('') : ''}
    `;

    document.getElementById('elderSettings').innerHTML = `
      <h3>给家人的 6 位连接数字</h3>
      <span class="code-box">${u.bind_code}</span>
      <p class="muted">家人输入后，申请会自动出现在主页，不需要点击“刷新”。</p>
      <h3>已经连接的人</h3>
      ${helpers.length ? helpers.map(helperHTML).join('') : '<p class="muted">暂时没有连接任何人。</p>'}
    `;

    document.getElementById('syncStatus').innerText = '守护状态已自动更新 · 不需要手动刷新';
    await loadElderRequests();
  } catch(e) {
    console.log(e);
    if (!silent) {
      document.getElementById('syncStatus').innerText = '网络暂时不稳定，系统稍后会自动重试';
    }
  } finally {
    elderRefreshBusy = false;
  }
}

async function unbindHelper(helperId) {
  if (!confirm('确认解除与这个人的连接吗？解除后，对方将收不到新的求助。')) return;
  try {
    await apiJSON('/api/unbind', {token: elderToken, helper_id: helperId});
    document.getElementById('helpStatus').innerHTML = '<p class="good">已解除连接。</p>';
    await refreshElder();
  } catch(e) { alert(e.message); }
}

async function createHelpRequest() {
  const button = document.getElementById('helpButton');
  if (button.disabled) {
    alert('还没有连接家人，请先让家人完成设置。');
    return;
  }
  const oldHTML = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '正在通知家人，请稍等……';
  document.getElementById('helpStatus').innerHTML = '';
  try {
    const data = await apiJSON('/api/create_help_request', {token: elderToken});
    latestRequestId = data.request.request_id;
    localStorage.setItem('gemmashield_latest_request_id', latestRequestId);
    document.getElementById('helpStatus').innerHTML = `
      <p class="good"><b>求助已经发出，已通知 ${data.notified_count} 位家人。</b><br>
      现在请不要转账，不要提供验证码、银行卡号或密码。需要时可在下方保留声音证据。</p>`;
    await loadElderRequests();
  } catch(e) {
    document.getElementById('helpStatus').innerHTML = `<p class="warning">求助发送失败：${e.message}</p>`;
  } finally {
    button.disabled = false;
    button.innerHTML = oldHTML;
  }
}

function requestHTML(r) {
  const q = r.quick_result;
  const g = window.gsGemmaResult(r);
  const simple = window.gsElderSimpleResult(r);
  const levelLabel = q ? window.gsQ(q, 'level_label') : '';
  const sourceLabel = q ? window.gsQ(q, 'evidence_source_label') : '';
  const confidenceLabel = q ? window.gsQ(q, 'evidence_confidence') : '';

  return `<div class="list-item">
    <b>${r.created_at}</b> <span class="badge">${r.status}</span><br>
    <span class="small">已通知：${(r.notified_helper_names || []).join('、') || '暂无'}</span>
    ${r.elder_audio_path ? '<p class="good">已保留声音证据。</p>' : ''}
    ${q ? `<p class="warning">${levelLabel}：${simple}</p><p class="small">证据来源：${sourceLabel || '未知'} | 证据置信度：${confidenceLabel || '中等'}</p>` : '<p class="muted">正在等待家人处理。</p>'}
    ${g ? `<pre data-no-i18n="true">${g.report}</pre>` : ''}
  </div>`;
}

async function loadElderRequests() {
  if (!elderToken) return;
  try {
    const res = await fetch('/api/elder_requests?token=' + encodeURIComponent(elderToken));
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'failed');
    document.getElementById('elderRequests').innerHTML = data.requests.length
      ? data.requests.map(requestHTML).join('')
      : '<p class="muted">暂无求助记录。</p>';
  } catch(e) {
    console.log(e);
  }
}

async function recordEvidence() {
  if (!latestRequestId) {
    alert('请先点击上面的红色求助按钮，再开始录音。');
    return;
  }
  const status = document.getElementById('recordStatus');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const form = new FormData();
      form.append('token', elderToken);
      form.append('request_id', latestRequestId);
      form.append('source_type', 'elder_recording');
      form.append('audio', blob, 'elder_evidence.webm');
      status.innerText = '正在上传录音……';
      const res = await fetch('/api/upload_evidence', {method:'POST', body: form});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      status.innerText = '录音已经保存，家人可以查看并处理。';
      await loadElderRequests();
    };
    recorder.start();
    status.innerText = '正在录制 12 秒。不要说出验证码、密码或银行卡号。';
    setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, 12000);
  } catch(e) { alert('录音失败：' + e.message); }
}

async function uploadEvidenceFile() {
  if (!latestRequestId) {
    alert('请先点击上面的红色求助按钮，再上传语音。');
    return;
  }
  const file = document.getElementById('elderEvidenceFile').files[0];
  if (!file) return alert('请先选择一段语音。');
  const sourceType = document.getElementById('elderUploadSource').value || 'elder_upload';
  const status = document.getElementById('recordStatus');
  try {
    const form = new FormData();
    form.append('token', elderToken);
    form.append('request_id', latestRequestId);
    form.append('source_type', sourceType);
    form.append('audio', file);
    status.innerText = '正在上传语音……';
    const res = await fetch('/api/upload_evidence', {method:'POST', body: form});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    status.innerText = '语音已经保存，家人可以查看并处理。';
    await loadElderRequests();
  } catch(e) { alert('上传失败：' + e.message); }
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden && elderToken) refreshElder(true);
});

if (elderToken) startElderAutoRefresh();
else showLogin();
</script>
""")

HELPER_HTML = page_shell("GemmaShield Helper", r"""
<div class="hero">
  <h1>👨‍👩‍👧 家属/社区端：可信协助人网络</h1>
  <p>家属或社区工作人员输入老人端显示的 6 位连接数字并提交申请。老人确认后，求助会进入你的收件箱。</p>
</div>

<div class="grid">
  <div class="card" id="loginCard">
    <h2>手机号登录 / 注册协助人账号</h2>
    <input class="input" id="helperName" placeholder="姓名/身份，例如：女儿" value="女儿" />
    <select id="relationship" class="input">
      <option value="family">家属 family</option>
      <option value="community">社区工作人员 community</option>
      <option value="volunteer">志愿者 volunteer</option>
    </select>
    <input class="input" id="helperPhone" placeholder="手机号，例如：18800000002" value="18800000002" />
    <button onclick="sendOtp('helper')">发送短信验证码</button>
    <div id="otpNotice" class="small"></div>
    <input class="input" id="helperOtp" placeholder="输入验证码" />
    <button class="goodbtn" onclick="verifyOtp('helper')">登录协助人端</button>
    <p class="muted">HF Demo 中验证码会显示在页面上；真实上线时接入短信服务商。</p>
  </div>

  <div class="card hidden" id="bindCard">
    <h2>绑定老人</h2>
    <div id="helperProfile"></div>
    <input class="input" id="bindCode" inputmode="numeric" maxlength="7" placeholder="输入老人端的 6 位数字，例如 438216" />
    <button onclick="bindElder()">发送连接申请</button>
    <button onclick="refreshHelper()">刷新</button>
  </div>
</div>

<div class="grid">
  <div class="card hidden" id="inboxCard">
    <h2>我的待处理请求</h2>
    <div id="inbox"></div>
  </div>

  <div class="card hidden" id="analysisCard">
    <h2>两阶段风险分析</h2>
    <div id="selectedReq" class="muted">请选择一个请求。</div>
    <div class="audio-box">
      <h3>第一阶段：CNN + Whisper</h3>
      <p class="muted">可以使用老人端保留证据，也可以由家属/社区端上传通话录音、语音留言或社交软件语音。系统会记录证据来源，并在风险结果中给出相应的核实提醒。</p>
      <select id="helperUploadSource" class="input">
        <option value="helper_upload">家属/社区端上传音频</option>
        <option value="call_recording">保存的通话录音</option>
        <option value="wechat_voice">微信/社交软件语音</option>
        <option value="voicemail">语音留言</option>
      </select>
      <input type="file" id="audioFile" class="input" accept="audio/*" />
      <button onclick="quickScanWithUpload()">上传音频并执行 CNN + Whisper</button>
      <button class="secondary" onclick="quickScanEvidence()">使用老人端保留证据执行 CNN + Whisper</button>
      <div id="quickResult"></div>
    </div>
    <div class="audio-box">
      <h3>第二阶段：Gemma 4 API 风险推理</h3>
      <p class="muted">Gemma 4 综合 CNN 声学证据、Whisper 文本证据和诈骗行为信号生成老人友好的建议。</p>
      <button class="goodbtn" onclick="gemmaReason()">执行 Gemma 4 风险推理</button>
      <div id="gemmaResult"></div>
    </div>
  </div>
</div>

<script>
let helperToken = localStorage.getItem('gemmashield_helper_token') || '';
let selectedRequestId = '';

async function apiJSON(url, data) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
  const json = await res.json();
  if (!res.ok) throw new Error(json.detail || JSON.stringify(json));
  return json;
}

async function sendOtp(role) {
  const phone = document.getElementById('helperPhone').value;
  const display_name = document.getElementById('helperName').value || '协助人';
  const relationship = document.getElementById('relationship').value;
  try {
    const data = await apiJSON('/api/send_otp', {phone, role, display_name, relationship});
    document.getElementById('otpNotice').innerHTML = `Demo 验证码：<b>${data.demo_otp}</b>（真实上线时通过短信发送）`;
    document.getElementById('helperOtp').value = data.demo_otp;
  } catch(e) { alert(e.message); }
}

async function verifyOtp(role) {
  const phone = document.getElementById('helperPhone').value;
  const display_name = document.getElementById('helperName').value || '协助人';
  const relationship = document.getElementById('relationship').value;
  const code = document.getElementById('helperOtp').value;
  try {
    const data = await apiJSON('/api/verify_otp', {phone, role, display_name, relationship, code});
    helperToken = data.token;
    localStorage.setItem('gemmashield_helper_token', helperToken);
    await refreshHelper();
  } catch(e) { alert(e.message); }
}

function elderHTML(e) {
  return `<div class="list-item"><b>${e.display_name}</b><br><span class="small">${e.phone}</span><button class="danger" onclick="unbindElder('${e.user_id}')">解绑此老人</button></div>`;
}

function pendingElderHTML(p) {
  return `<div class="list-item"><b>${p.elder_name}</b> <span class="badge">等待老人确认</span><br><span class="small">${p.elder_phone} | 申请时间：${p.created_at || ''}</span><p class="muted">绑定申请已发送到老人端。老人确认后，你才会正式成为协助人。</p></div>`;
}

async function refreshHelper() {
  if (!helperToken) return;
  try {
    const res = await fetch('/api/me?token=' + encodeURIComponent(helperToken));
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '登录失效');
    document.getElementById('loginCard').classList.add('hidden');
    document.getElementById('bindCard').classList.remove('hidden');
    document.getElementById('inboxCard').classList.remove('hidden');
    document.getElementById('analysisCard').classList.remove('hidden');
    const u = data.user;
    const elders = u.bound_elders || [];
    const pending = u.pending_binding_requests || [];
    document.getElementById('helperProfile').innerHTML = `
      <p><b>当前协助人：</b>${u.display_name} <span class="badge">${u.relationship}</span></p>
      <p><b>已绑定老人：</b>${elders.length}</p>
      <div>${elders.length ? elders.map(elderHTML).join('') : '<p class="warning">还没有正式绑定老人。请输入老人端显示的家庭守护绑定码，并等待老人端确认。</p>'}</div>
      <h3>等待老人确认</h3>
      <div>${pending.length ? pending.map(pendingElderHTML).join('') : '<p class="muted">暂无待确认绑定申请。</p>'}</div>
    `;
    await loadInbox();
  } catch(e) {
    console.log(e);
    localStorage.removeItem('gemmashield_helper_token');
  }
}

async function bindElder() {
  const bind_code = document.getElementById('bindCode').value.replace(/\D/g, '');
  try {
    const data = await apiJSON('/api/bind_elder', {token: helperToken, bind_code});
    alert(data.message);
    document.getElementById('bindCode').value = '';
    await refreshHelper();
  } catch(e) { alert(e.message); }
}

async function unbindElder(elderId) {
  if (!confirm('确认解除与这个老人的绑定吗？')) return;
  try {
    await apiJSON('/api/unbind', {token: helperToken, elder_id: elderId});
    await refreshHelper();
  } catch(e) { alert(e.message); }
}

function reqHTML(r) {
  const q = r.quick_result;
  const levelLabel = q ? window.gsQ(q, 'level_label') : '';
  return `<div class="list-item">
    <b>${r.elder_name}</b> 发起求助 <span class="badge">${r.status}</span><br>
    <span class="small">${r.request_id} | ${r.created_at}</span><br>
    ${r.elder_audio_path ? '<span class="badge">有老人端证据</span>' : '<span class="badge">等待音频</span>'}
    ${q ? `<span class="badge">${levelLabel}</span>` : ''}
    <button onclick="selectRequest('${r.request_id}')">选择处理</button>
  </div>`;
}

async function loadInbox() {
  const res = await fetch('/api/helper_inbox?token=' + encodeURIComponent(helperToken));
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'failed');
  document.getElementById('inbox').innerHTML = data.requests.length ? data.requests.map(reqHTML).join('') : '<p class="muted">暂无待处理请求。</p>';
}

async function selectRequest(id) {
  selectedRequestId = id;
  document.getElementById('selectedReq').innerHTML = `<b>当前处理请求：</b><span class="badge">${id}</span>`;
  document.getElementById('quickResult').innerHTML = '';
  document.getElementById('gemmaResult').innerHTML = '';
  await loadRequestDetail();
}

async function loadRequestDetail() {
  if (!selectedRequestId) return;
  const res = await fetch('/api/request_status?token=' + encodeURIComponent(helperToken) + '&request_id=' + encodeURIComponent(selectedRequestId));
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'failed');
  renderResults(data.request);
}

function renderResults(r) {
  if (r.quick_result) {
    const q = r.quick_result;
    const levelLabel = window.gsQ(q, 'level_label');
    const sourceLabel = window.gsQ(q, 'evidence_source_label');
    const confidenceLabel = window.gsQ(q, 'evidence_confidence');
    const evidenceNote = window.gsQ(q, 'evidence_note');
    const scamLabel = window.gsQ(q, 'scam_label');
    const signalsLabel = window.gsQ(q, 'signals_label');

    document.getElementById('quickResult').innerHTML = `<div class="list-item">
      <h3>${levelLabel}</h3>
      <p><b>CNN 声学辅助证据：</b>AI伪造概率 ${q.fake.toFixed(2)}，真实语音概率 ${q.real.toFixed(2)}</p>
      <p><b>证据来源：</b>${sourceLabel || '未知'} | <b>证据置信度：</b>${confidenceLabel || '中等'}</p>
      <p class="muted"><b>证据使用提醒：</b>${evidenceNote || '请结合家属/社区人员核实。'}</p>
      <p><b>Whisper 转录：</b><span data-no-i18n="true">${q.transcript}</span></p>
      <p><b>风险类型：</b>${scamLabel}</p>
      <p><b>风险信号：</b>${signalsLabel}</p>
      <p class="small">处理时间：${q.processing_seconds} 秒</p>
    </div>`;
  }

  const g = window.gsGemmaResult(r);
  if (g) {
    document.getElementById('gemmaResult').innerHTML = `<div class="list-item"><h3>Gemma 4 风险推理</h3><p class="small">来源：${g.source}</p><pre data-no-i18n="true">${g.report}</pre></div>`;
  } else {
    document.getElementById('gemmaResult').innerHTML = '';
  }
}

async function quickScanWithUpload() {
  if (!selectedRequestId) return alert('请先选择一个请求。');
  const file = document.getElementById('audioFile').files[0];
  if (!file) return alert('请先选择音频文件。');
  const form = new FormData();
  form.append('token', helperToken);
  form.append('request_id', selectedRequestId);
  form.append('source_type', document.getElementById('helperUploadSource').value || 'helper_upload');
  form.append('audio', file);
  document.getElementById('quickResult').innerHTML = '<p class="warning">正在执行 CNN + Whisper，请稍等...</p>';
  try {
    const res = await fetch('/api/quick_scan', {method:'POST', body: form});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    renderResults(data.request);
    await loadInbox();
  } catch(e) { alert(e.message); }
}

async function quickScanEvidence() {
  if (!selectedRequestId) return alert('请先选择一个请求。');
  const form = new FormData();
  form.append('token', helperToken);
  form.append('request_id', selectedRequestId);
  document.getElementById('quickResult').innerHTML = '<p class="warning">正在使用老人端证据执行 CNN + Whisper，请稍等...</p>';
  try {
    const res = await fetch('/api/quick_scan', {method:'POST', body: form});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    renderResults(data.request);
    await loadInbox();
  } catch(e) { alert(e.message); }
}

async function gemmaReason() {
  if (!selectedRequestId) return alert('请先选择一个请求。');
  document.getElementById('gemmaResult').innerHTML = '<p class="warning">正在调用 Gemma 4 API 生成风险推理...</p>';
  try {
    const data = await apiJSON('/api/gemma_reason', {token: helperToken, request_id: selectedRequestId});
    renderResults(data.request);
    await loadInbox();
  } catch(e) { alert(e.message); }
}

setInterval(() => { if (helperToken) loadInbox(); }, 5000);
refreshHelper();
</script>
""")

# =========================================================
# Routes: pages
# =========================================================


@app.get("/", response_class=HTMLResponse)
def root():
    return ROOT_HTML


@app.get("/elder", response_class=HTMLResponse)
def elder_page():
    return ELDER_HTML


@app.get("/helper", response_class=HTMLResponse)
def helper_page():
    return HELPER_HTML


@app.get("/health")
def health():
    db = load_db()
    return {
        "ok": True,
        "app": APP_NAME,
        "time": now_str(),
        "users": len(db.get("users", {})),
        "requests": len(db.get("requests", {})),
        "gemma_api_configured": bool(HF_TOKEN),
        "sms_provider_enabled": SMS_PROVIDER_ENABLED,
        "wechat_provider_enabled": WECHAT_PROVIDER_ENABLED,
        "push_provider_enabled": PUSH_PROVIDER_ENABLED,
    }

# =========================================================
# Routes: auth / binding
# =========================================================


@app.post("/api/send_otp")
def send_otp(payload: Dict[str, Any]):
    phone = normalize_phone(payload.get("phone", ""))
    role = payload.get("role", "")
    display_name = str(payload.get("display_name") or "用户").strip()
    if role not in ["elder", "helper"]:
        raise HTTPException(status_code=400, detail="role must be elder or helper")
    if len(phone) < 6:
        raise HTTPException(status_code=400, detail="请输入有效手机号")

    code = f"{random.randint(100000, 999999)}"
    db = load_db()
    db["otps"][phone] = {
        "code": code,
        "role": role,
        "display_name": display_name,
        "relationship": payload.get("relationship", "family"),
        "created_at": time.time(),
        "expires_at": time.time() + 10 * 60,
        "channel": "sms",
        "status": "demo_code_displayed" if not SMS_PROVIDER_ENABLED else "provider_hook_ready",
    }
    save_db(db)
    # For demo only: return code visibly. Real SMS should not return this to frontend.
    return {
        "ok": True,
        "message": "验证码已生成。HF Demo 中直接显示验证码；真实上线时由短信服务商发送。",
        "demo_otp": code,
        "sms_provider_enabled": SMS_PROVIDER_ENABLED,
    }


@app.post("/api/verify_otp")
def verify_otp(payload: Dict[str, Any]):
    phone = normalize_phone(payload.get("phone", ""))
    code = str(payload.get("code", "")).strip()
    role = payload.get("role", "")
    display_name = str(payload.get("display_name") or "用户").strip()
    relationship = payload.get("relationship", "family")
    if role not in ["elder", "helper"]:
        raise HTTPException(status_code=400, detail="role must be elder or helper")

    db = load_db()
    otp = db["otps"].get(phone)
    if not otp:
        raise HTTPException(status_code=400, detail="请先发送验证码")
    if time.time() > otp.get("expires_at", 0):
        raise HTTPException(status_code=400, detail="验证码已过期，请重新发送")
    if code != otp.get("code"):
        raise HTTPException(status_code=400, detail="验证码错误")

    existing_uid = db["phone_index"].get(phone)
    if existing_uid:
        user = db["users"].get(existing_uid)
        if user and user.get("role") != role:
            raise HTTPException(status_code=400, detail="该手机号已注册为另一种角色。Demo 中请使用另一个手机号。")
        if not user:
            existing_uid = None

    if existing_uid:
        uid = existing_uid
        user = db["users"][uid]
        user["display_name"] = display_name or user.get("display_name", "用户")
        if role == "helper":
            user["relationship"] = relationship
    else:
        uid = new_id("U")
        user = {
            "user_id": uid,
            "phone": phone,
            "role": role,
            "display_name": display_name,
            "created_at": now_str(),
            "session_tokens": [],
        }
        if role == "elder":
            user["bind_code"] = make_bind_code(uid)
            user["bound_helpers"] = []
        else:
            user["bound_elders"] = []
            user["relationship"] = relationship
            user["channels"] = ["sms", "wechat", "push"]
        db["users"][uid] = user
        db["phone_index"][phone] = uid

    token = secrets.token_urlsafe(24)
    user.setdefault("session_tokens", []).append(token)
    # keep tokens limited
    user["session_tokens"] = user["session_tokens"][-5:]
    db["otps"].pop(phone, None)
    save_db(db)
    return {"ok": True, "token": token, "user": user_public(user, db)}


@app.get("/api/me")
def me(token: str = Query(...)):
    user = auth_user(token)
    db = load_db()
    return {"ok": True, "user": user_public(user, db)}


@app.post("/api/bind_elder")
def bind_elder(payload: Dict[str, Any]):
    """
    Helper enters elder binding code.

    New safer flow:
    1) Helper submits elder bind_code.
    2) System creates a pending binding request.
    3) Elder must confirm on elder side.
    4) Only after elder approval do we write both directions:
       elder.bound_helpers and helper.bound_elders.
    """
    helper_auth = auth_user(payload.get("token"), role="helper")
    raw_bind_code = str(payload.get("bind_code") or "").strip()
    bind_code = re.sub(r"\D", "", raw_bind_code) if not re.search(r"[A-Za-z]", raw_bind_code) else raw_bind_code.upper()
    if not bind_code:
        raise HTTPException(status_code=400, detail="请输入老人端显示的 6 位连接数字")
    if bind_code.isdigit() and len(bind_code) != 6:
        raise HTTPException(status_code=400, detail="连接数字应为 6 位")

    db = load_db()
    helper = db["users"].get(helper_auth["user_id"])
    if not helper or helper.get("role") != "helper":
        raise HTTPException(status_code=401, detail="协助人账号不存在，请重新登录")

    elder = None
    for u in db["users"].values():
        if u.get("role") == "elder" and str(u.get("bind_code", "")).upper() == bind_code:
            elder = u
            break
    if elder is None:
        raise HTTPException(status_code=404, detail="没有找到该绑定码对应的老人账号")

    helper_id = helper["user_id"]
    elder_id = elder["user_id"]
    elder.setdefault("bound_helpers", [])
    helper.setdefault("bound_elders", [])

    already_bound = helper_id in elder.get("bound_helpers", []) or elder_id in helper.get("bound_elders", [])
    if already_bound:
        if helper_id not in elder["bound_helpers"] and len(elder["bound_helpers"]) < MAX_HELPERS_PER_ELDER:
            elder["bound_helpers"].append(helper_id)
        if elder_id not in helper["bound_elders"]:
            helper["bound_elders"].append(elder_id)
        save_db(db)
        return {
            "ok": True,
            "status": "already_bound",
            "message": f"你已绑定 {elder.get('display_name')}。",
            "elder": user_public(elder, db),
            "helper": user_public(helper, db),
        }

    if len(elder.get("bound_helpers", [])) >= MAX_HELPERS_PER_ELDER:
        raise HTTPException(status_code=400, detail=f"该老人已绑定 {MAX_HELPERS_PER_ELDER} 位协助人，不能继续绑定")

    # Reuse existing pending request if helper has already applied.
    for bid, br in db.get("binding_requests", {}).items():
        if (br.get("elder_id") == elder_id and br.get("helper_id") == helper_id and br.get("status") == "pending"):
            return {
                "ok": True,
                "status": "pending",
                "binding_request_id": bid,
                "message": f"绑定申请已发送给 {elder.get('display_name')}，请等待老人端确认。",
                "elder": user_public(elder, db),
                "helper": user_public(helper, db),
            }

    bid = new_id("BR")
    br = {
        "binding_request_id": bid,
        "elder_id": elder_id,
        "elder_name": elder.get("display_name", "老人"),
        "helper_id": helper_id,
        "helper_name": helper.get("display_name", "协助人"),
        "helper_phone": helper.get("phone", ""),
        "relationship": helper.get("relationship", "family"),
        "bind_code": bind_code,
        "status": "pending",
        "created_at": now_str(),
        "decided_at": None,
    }
    db.setdefault("binding_requests", {})[bid] = br
    save_db(db)
    return {
        "ok": True,
        "status": "pending",
        "binding_request_id": bid,
        "message": f"绑定申请已发送到 {elder.get('display_name')} 的老人端。老人确认后，你才会正式绑定。",
        "elder": user_public(elder, db),
        "helper": user_public(helper, db),
    }


@app.post("/api/confirm_binding")
def confirm_binding(payload: Dict[str, Any]):
    elder_auth = auth_user(payload.get("token"), role="elder")
    binding_request_id = str(payload.get("binding_request_id") or "").strip()
    action = str(payload.get("action") or "").strip().lower()
    if action not in ["approve", "reject"]:
        raise HTTPException(status_code=400, detail="action must be approve or reject")

    db = load_db()
    elder = db["users"].get(elder_auth["user_id"])
    br = db.get("binding_requests", {}).get(binding_request_id)
    if not br:
        raise HTTPException(status_code=404, detail="绑定申请不存在")
    if br.get("elder_id") != elder.get("user_id"):
        raise HTTPException(status_code=403, detail="你无权处理这个绑定申请")
    if br.get("status") != "pending":
        return {"ok": True, "message": f"该申请已处理：{br.get('status')}", "user": user_public(elder, db)}

    helper = db["users"].get(br.get("helper_id"))
    if not helper or helper.get("role") != "helper":
        br["status"] = "rejected"
        br["decided_at"] = now_str()
        save_db(db)
        raise HTTPException(status_code=404, detail="申请人账号不存在，已自动拒绝")

    elder.setdefault("bound_helpers", [])
    helper.setdefault("bound_elders", [])

    if action == "reject":
        br["status"] = "rejected"
        br["decided_at"] = now_str()
        save_db(db)
        return {"ok": True, "message": "已拒绝该绑定申请。", "user": user_public(elder, db)}

    if helper["user_id"] not in elder["bound_helpers"]:
        if len(elder["bound_helpers"]) >= MAX_HELPERS_PER_ELDER:
            raise HTTPException(status_code=400, detail=f"该老人已绑定 {MAX_HELPERS_PER_ELDER} 位协助人，不能继续绑定")
        elder["bound_helpers"].append(helper["user_id"])
    if elder["user_id"] not in helper["bound_elders"]:
        helper["bound_elders"].append(elder["user_id"])

    br["status"] = "approved"
    br["decided_at"] = now_str()
    save_db(db)
    return {
        "ok": True,
        "message": f"已确认：{helper.get('display_name')} 成为你的可信协助人。",
        "user": user_public(elder, db),
        "helper": user_public(helper, db),
    }


@app.post("/api/unbind")
def unbind(payload: Dict[str, Any]):
    user = auth_user(payload.get("token"))
    db = load_db()
    if user.get("role") == "elder":
        helper_id = payload.get("helper_id")
        if not helper_id:
            raise HTTPException(status_code=400, detail="缺少 helper_id")
        elder = db["users"].get(user["user_id"])
        helper = db["users"].get(helper_id)
        if helper_id in elder.get("bound_helpers", []):
            elder["bound_helpers"].remove(helper_id)
        if helper and elder["user_id"] in helper.get("bound_elders", []):
            helper["bound_elders"].remove(elder["user_id"])
        for br in db.get("binding_requests", {}).values():
            if br.get("elder_id") == elder["user_id"] and br.get("helper_id") == helper_id and br.get("status") == "pending":
                br["status"] = "cancelled"
                br["decided_at"] = now_str()
        save_db(db)
        return {"ok": True, "message": "已解绑该协助人"}

    if user.get("role") == "helper":
        elder_id = payload.get("elder_id")
        if not elder_id:
            raise HTTPException(status_code=400, detail="缺少 elder_id")
        helper = db["users"].get(user["user_id"])
        elder = db["users"].get(elder_id)
        if elder_id in helper.get("bound_elders", []):
            helper["bound_elders"].remove(elder_id)
        if elder and helper["user_id"] in elder.get("bound_helpers", []):
            elder["bound_helpers"].remove(helper["user_id"])
        for br in db.get("binding_requests", {}).values():
            if br.get("elder_id") == elder_id and br.get("helper_id") == helper["user_id"] and br.get("status") == "pending":
                br["status"] = "cancelled"
                br["decided_at"] = now_str()
        save_db(db)
        return {"ok": True, "message": "已解除与该老人的绑定"}

    raise HTTPException(status_code=403, detail="Invalid role")

# =========================================================
# Routes: requests / evidence / analysis
# =========================================================


def enrich_request(req: Dict[str, Any], db: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(req)
    elder = db["users"].get(r.get("elder_id"), {})
    r["elder_name"] = elder.get("display_name", "老人")
    names = []
    for hid in r.get("notified_helpers", []):
        h = db["users"].get(hid)
        if h:
            names.append(h.get("display_name", "协助人"))
    r["notified_helper_names"] = names
    return r


@app.post("/api/create_help_request")
def create_help_request(payload: Dict[str, Any]):
    elder = auth_user(payload.get("token"), role="elder")
    db = load_db()
    elder = db["users"][elder["user_id"]]
    helper_ids = list(elder.get("bound_helpers", []))

    rid = f"GS-{int(time.time())}-{uuid.uuid4().hex[:4].upper()}"
    request = {
        "request_id": rid,
        "elder_id": elder["user_id"],
        "notified_helpers": helper_ids,
        "status": "waiting_for_audio_or_scan" if helper_ids else "no_bound_helper",
        "created_at": now_str(),
        "elder_audio_path": None,
        "helper_audio_path": None,
        "audio_source": None,
        "quick_result": None,
        "gemma_result": None,
        "gemma_result_zh": None,
        "gemma_result_en": None,
        "elder_simple_result": None,
        "elder_simple_result_zh": None,
        "elder_simple_result_en": None,
        "notification_ids": [],
    }
    sent = dispatch_notifications(db, rid, elder, helper_ids)
    request["notification_ids"] = [n["notification_id"] for n in sent]
    db["requests"][rid] = request
    save_db(db)
    return {
        "ok": True,
        "request": enrich_request(request, db),
        "notified_count": len(helper_ids),
        "notifications": sent,
        "message": f"已通知 {len(helper_ids)} 位可信协助人。" if helper_ids else "还没有绑定协助人，请先绑定。",
    }


@app.get("/api/elder_requests")
def elder_requests(token: str = Query(...)):
    elder = auth_user(token, role="elder")
    db = load_db()
    items = [enrich_request(r, db) for r in db["requests"].values() if r.get("elder_id") == elder["user_id"]]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"ok": True, "requests": items}


@app.get("/api/helper_inbox")
def helper_inbox(token: str = Query(...)):
    helper = auth_user(token, role="helper")
    db = load_db()
    hid = helper["user_id"]
    items = [enrich_request(r, db) for r in db["requests"].values() if hid in r.get("notified_helpers", [])]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"ok": True, "requests": items}


@app.get("/api/request_status")
def request_status(token: str = Query(...), request_id: str = Query(...)):
    user = auth_user(token)
    db = load_db()
    req = db["requests"].get(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="请求不存在")
    if not can_access_request(user, req):
        raise HTTPException(status_code=403, detail="你无权查看该请求")
    return {"ok": True, "request": enrich_request(req, db)}


def save_upload_file(upload: UploadFile, prefix: str) -> str:
    suffix = Path(upload.filename or "audio.webm").suffix or ".webm"
    safe_name = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    dst = UPLOAD_DIR / safe_name
    with dst.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dst)


@app.post("/api/upload_evidence")
def upload_evidence(
    token: str = Form(...),
    request_id: str = Form(...),
    source_type: str = Form("elder_recording"),
    audio: UploadFile = File(...),
):
    user = auth_user(token)
    db = load_db()
    req = db["requests"].get(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="请求不存在")
    if not can_access_request(user, req):
        raise HTTPException(status_code=403, detail="你无权上传该请求证据")
    source = normalize_evidence_source(source_type)
    path = save_upload_file(audio, source)
    if user.get("role") == "elder":
        req["elder_audio_path"] = path
        req["elder_audio_source"] = source
        req["audio_source"] = source
    else:
        req["helper_audio_path"] = path
        req["helper_audio_source"] = source
        req["audio_source"] = source
    req["status"] = "audio_ready_waiting_for_scan"
    req["updated_at"] = now_str()
    save_db(db)
    return {"ok": True, "request": enrich_request(req, db)}


@app.post("/api/quick_scan")
def quick_scan(
    token: str = Form(...),
    request_id: str = Form(...),
    source_type: str = Form(""),
    lang: str = Form("zh"),
    audio: Optional[UploadFile] = File(None),
):
    user = auth_user(token)
    db = load_db()
    req = db["requests"].get(request_id)
    if not req:
        raise HTTPException(status_code=404, detail="请求不存在")
    if not can_access_request(user, req):
        raise HTTPException(status_code=403, detail="你无权处理该请求")

    if audio is not None:
        source = normalize_evidence_source(source_type or "helper_upload")
        path = save_upload_file(audio, source)
        req["helper_audio_path"] = path
        req["helper_audio_source"] = source
        req["audio_source"] = source
    else:
        path = req.get("elder_audio_path") or req.get("helper_audio_path")
        if req.get("elder_audio_path") and path == req.get("elder_audio_path"):
            source = normalize_evidence_source(req.get("elder_audio_source") or req.get("audio_source") or "elder_recording")
        else:
            source = normalize_evidence_source(req.get("helper_audio_source") or req.get("audio_source") or "helper_upload")

    if not path:
        raise HTTPException(status_code=400, detail="没有可用音频。请上传录音，或先让老人端保留证据。")

    try:
        quick = quick_scan_audio(path, source)
    except Exception as e:
        print("quick_scan failed:", repr(e))
        raise HTTPException(status_code=500, detail=f"CNN + Whisper 扫描失败：{repr(e)}")

    lang = normalize_lang(lang)
    req["quick_result"] = quick
    req["status"] = "quick_scanned_waiting_for_gemma"

    # Keep both languages so switching the UI never requires re-running the CNN.
    req["elder_simple_result_zh"] = elder_simple_notice(quick["level"], "zh")
    req["elder_simple_result_en"] = elder_simple_notice(quick["level"], "en")
    req["elder_simple_result"] = (
        req["elder_simple_result_en"] if lang == "en" else req["elder_simple_result_zh"]
    )
    req["updated_at"] = now_str()
    db["requests"][request_id] = req
    save_db(db)
    return {"ok": True, "request": enrich_request(req, db)}


@app.post("/api/gemma_reason")
def gemma_reason(payload: Dict[str, Any]):
    user = auth_user(payload.get("token"))
    request_id = payload.get("request_id")
    lang = normalize_lang(payload.get("lang", "zh"))

    db = load_db()
    req = db["requests"].get(request_id)
    if not req:
        raise HTTPException(status_code=404, detail=localized(lang, "请求不存在", "Request not found"))
    if not can_access_request(user, req):
        raise HTTPException(
            status_code=403,
            detail=localized(lang, "你无权处理该请求", "You do not have permission to process this request"),
        )
    if not req.get("quick_result"):
        raise HTTPException(
            status_code=400,
            detail=localized(
                lang,
                "请先完成第一阶段 CNN + Whisper 扫描",
                "Please complete Stage 1 CNN + Whisper analysis first",
            ),
        )

    elder = db["users"].get(req.get("elder_id"), {})
    elder_name = elder.get("display_name", "老人" if lang == "zh" else "Older adult")
    result = analyze_with_gemma(req["quick_result"], elder_name, lang)

    gemma_payload = {
        "source": result.get("source"),
        "report": result.get("report"),
        "error": result.get("error"),
        "lang": lang,
        "created_at": now_str(),
    }

    # Preserve the latest result for backward compatibility and also keep a
    # language-specific copy so switching the UI never shows the wrong-language report.
    req["gemma_result"] = gemma_payload
    req[f"gemma_result_{lang}"] = gemma_payload

    req["status"] = "completed"
    req["updated_at"] = now_str()
    req["elder_simple_result_zh"] = elder_simple_notice(
        req["quick_result"].get("level", "MEDIUM"), "zh"
    )
    req["elder_simple_result_en"] = elder_simple_notice(
        req["quick_result"].get("level", "MEDIUM"), "en"
    )
    req["elder_simple_result"] = (
        req["elder_simple_result_en"] if lang == "en" else req["elder_simple_result_zh"]
    )

    db["requests"][request_id] = req
    save_db(db)
    return {"ok": True, "request": enrich_request(req, db)}

# =========================================================
# Dev/demo utility routes
# =========================================================


@app.post("/api/logout_all")
def logout_all(payload: Dict[str, Any]):
    user = auth_user(payload.get("token"))
    db = load_db()
    u = db["users"].get(user["user_id"])
    if u:
        u["session_tokens"] = []
        save_db(db)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
