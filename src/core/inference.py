import torch
import torch.nn.functional as F
import logging
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from src.core.models.asteroid_separator import AsteroidSeparator

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    16kHz Dual-Engine Scrubber.
    Asteroid for separation + DNS64 for artifact scrubbing + TitaNet for Smart Picking.
    """
    def __init__(self, separator_type="asteroid"):
        separator_factory = {
            "asteroid": AsteroidSeparator,
        }
        if separator_type not in separator_factory:
            raise ValueError(f"Unknown separator type: {separator_type}")
            
        logger.info(f"Initializing {separator_type.capitalize()} BSS Engine...")
        self.separator = separator_factory[separator_type]()
        
        logger.info("Initializing TitaNet Smart Picker...")
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").cuda().eval()
        
        logger.info("Initializing DNS64 Artifact Scrubber...")
        self.cleaner = torch.hub.load('facebookresearch/denoiser', 'dns64').cuda().eval()
        
        # Sticky Picker Trackers (Task 26)
        self.ema_score_0 = 0.0
        self.ema_score_1 = 0.0
        self.is_fading_in = True 
        
        # Hang-Time Holdover (Task 27)
        self.hang_time_frames = 0
        self.MAX_HANG_TIME = 6  # 190ms holdover to decapitate replies
        self.locked_channel = 0  # NEW: Tracks the last valid target channel
        
        # Warmup all models for zero-jitter streaming
        with torch.no_grad():
            dummy_cuda = torch.randn(1, 1, 24000).cuda()
            _ = self.separator.separate(dummy_cuda)
            _ = self.cleaner(dummy_cuda)
            
        logger.info("Dual-Engine Extraction fully initialized.")

    def extract_voice(self, chunk_tensor: torch.Tensor, voiceprint_tensor: torch.Tensor) -> torch.Tensor:
        """
        Separates overlapping audio, picks target, and scrubs artifacts.
        """
        with torch.no_grad():
            # 1. Blind Source Separation (BSS)
            # sources shape: [1, 2, 24000]
            sources = self.separator.separate(chunk_tensor)
            source_0 = sources[:, 0, :]
            source_1 = sources[:, 1, :]
            
            # 2. Smart Picking Score
            audio_len = torch.tensor([source_0.shape[1]]).cuda()
            _, emb_0 = self.speaker_model.forward(input_signal=source_0, input_signal_length=audio_len)
            _, emb_1 = self.speaker_model.forward(input_signal=source_1, input_signal_length=audio_len)
            
            score_0 = F.cosine_similarity(emb_0, voiceprint_tensor).mean().item()
            score_1 = F.cosine_similarity(emb_1, voiceprint_tensor).mean().item()
            
            # 3. Peak Asymmetric EMA
            self.ema_score_0 = score_0 if score_0 > self.ema_score_0 else (0.80 * self.ema_score_0 + 0.20 * score_0)
            self.ema_score_1 = score_1 if score_1 > self.ema_score_1 else (0.80 * self.ema_score_1 + 0.20 * score_1)
            
            # 4 & 5. SINGLE-THRESHOLD CHANNEL LOCK (Peak Configuration)
            if self.ema_score_0 >= self.ema_score_1:
                current_best_score, current_best_channel = self.ema_score_0, 0
            else:
                current_best_score, current_best_channel = self.ema_score_1, 1
                
            CONFIDENCE_THRESHOLD = 0.30 
            
            if current_best_score >= CONFIDENCE_THRESHOLD:
                self.locked_channel = current_best_channel
                self.hang_time_frames = self.MAX_HANG_TIME
                winner = source_0 if self.locked_channel == 0 else source_1
            else:
                self.hang_time_frames -= 1
                if self.hang_time_frames > 0:
                    winner = source_0 if self.locked_channel == 0 else source_1
                else:
                    self.hang_time_frames = 0
                    self.is_fading_in = True 
                    return torch.zeros(1, 1, source_0.shape[-1], device=source_0.device)
            
            # 6. The Scrubber
            cleaned_audio = self.cleaner(winner.unsqueeze(1))
            
            # 7. ANTI-POP LINEAR FADE-IN
            if getattr(self, 'is_fading_in', False):
                fade_curve = torch.linspace(0.0, 1.0, steps=cleaned_audio.shape[-1], device=cleaned_audio.device)
                cleaned_audio = cleaned_audio * fade_curve
                self.is_fading_in = False
                
            return cleaned_audio
