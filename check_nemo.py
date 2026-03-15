try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    print("NEMO_READY")
except ImportError as e:
    print(f"NEMO_FAILED: {e}")
