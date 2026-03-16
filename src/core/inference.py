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
        
        # Multi-Target Trackers
        self.ema_scores = {} # {speaker_id: {0: score, 1: score}}
        self.hang_time_frames = {}
        self.locked_channel = {}
        self.is_fading_in = {}
        self.MAX_HANG_TIME = 6 
        
        # Warmup all models for zero-jitter streaming
        with torch.no_grad():
            dummy_cuda = torch.randn(1, 1, 24000).cuda()
            _ = self.separator.separate(dummy_cuda)
            _ = self.cleaner(dummy_cuda)
            
        logger.info("Dual-Engine Extraction fully initialized.")

    def extract_voices(self, chunk_tensor: torch.Tensor, target_profiles: dict) -> dict:
        """
        Separates overlapping audio, evaluates all target profiles, and returns a dict of extracted voices.
        """
        with torch.no_grad():
            # 1. Blind Source Separation (BSS)
            # sources shape: [1, 2, 24000]
            sources = self.separator.separate(chunk_tensor)
            source_0 = sources[:, 0, :]
            source_1 = sources[:, 1, :]
            
            # 2. Extract embeddings for both sources
            audio_len = torch.tensor([source_0.shape[1]]).cuda()
            _, emb_0 = self.speaker_model.forward(input_signal=source_0, input_signal_length=audio_len)
            _, emb_1 = self.speaker_model.forward(input_signal=source_1, input_signal_length=audio_len)
            
            outputs = {}
            for speaker_id, voiceprint_tensor in target_profiles.items():
                if speaker_id not in self.ema_scores:
                    self.ema_scores[speaker_id] = {0: 0.0, 1: 0.0}
                    self.hang_time_frames[speaker_id] = 0
                    self.locked_channel[speaker_id] = 0
                    self.is_fading_in[speaker_id] = True
                    
                # 3. Smart Picking Scores per speaker
                score_0 = F.cosine_similarity(emb_0, voiceprint_tensor).mean().item()
                score_1 = F.cosine_similarity(emb_1, voiceprint_tensor).mean().item()
                print(f"📊 [Scoring] {speaker_id} | Ch0: {score_0:.3f} | Ch1: {score_1:.3f}")
                
                # 4. Independent Asymmetric EMA per speaker
                self.ema_scores[speaker_id][0] = score_0 if score_0 > self.ema_scores[speaker_id][0] else (0.80 * self.ema_scores[speaker_id][0] + 0.20 * score_0)
                self.ema_scores[speaker_id][1] = score_1 if score_1 > self.ema_scores[speaker_id][1] else (0.80 * self.ema_scores[speaker_id][1] + 0.20 * score_1)
                
                # 5. Channel Lock per speaker
                if self.ema_scores[speaker_id][0] >= self.ema_scores[speaker_id][1]:
                    current_best_score, current_best_channel = self.ema_scores[speaker_id][0], 0
                else:
                    current_best_score, current_best_channel = self.ema_scores[speaker_id][1], 1
                    
                # Lowered to 0.10 to allow for room echo/speaker distortion matches
                CONFIDENCE_THRESHOLD = 0.10
                
                if current_best_score >= CONFIDENCE_THRESHOLD:
                    self.locked_channel[speaker_id] = current_best_channel
                    self.hang_time_frames[speaker_id] = self.MAX_HANG_TIME
                    winner = source_0 if self.locked_channel[speaker_id] == 0 else source_1
                else:
                    self.hang_time_frames[speaker_id] -= 1
                    if self.hang_time_frames[speaker_id] > 0:
                        winner = source_0 if self.locked_channel[speaker_id] == 0 else source_1
                    else:
                        self.hang_time_frames[speaker_id] = 0
                        self.is_fading_in[speaker_id] = True 
                        outputs[speaker_id] = torch.zeros(1, 1, source_0.shape[-1], device=source_0.device)
                        continue
                
                # 6. Scrub artifacts
                cleaned_audio = self.cleaner(winner.unsqueeze(1))
                
                # 7. Anti-pop Linear Fade-in per speaker
                if self.is_fading_in[speaker_id]:
                    fade_curve = torch.linspace(0.0, 1.0, steps=cleaned_audio.shape[-1], device=cleaned_audio.device)
                    cleaned_audio = cleaned_audio * fade_curve
                    self.is_fading_in[speaker_id] = False
                    
                outputs[speaker_id] = cleaned_audio
                
            return outputs
