from typing import Dict, List, Tuple, Union

import hydra
import omegaconf
import torch
from torch import Tensor, nn
from typing import Any

from .preprocess import SAMPLE_RATE, load_audio
from .utils import onnx_converter
import numpy as np

LONGFORM_THRESHOLD = 25 * SAMPLE_RATE


def _merge_word_timestamps(
    prev_words: List[Dict[str, Any]], new_words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merges two lists of word timestamp dictionaries using a fault-tolerant
    comparison of their underlying token IDs.

    It finds the longest sequence of words at the end of `prev_words` that
    best matches a sequence at the beginning of `new_words`.
    """
    if not prev_words:
        return new_words
    if not new_words:
        return prev_words

    # We'll compare based on the sequence of token IDs for each word
    prev_word_tokens = [word["token_ids"] for word in prev_words]
    new_word_tokens = [word["token_ids"] for word in new_words]

    best_overlap_n_words = 0
    max_matching_score = 0.0

    # Iterate through possible overlap lengths (in number of words)
    for k in range(1, min(len(prev_word_tokens), len(new_word_tokens)) + 1):
        # The last k words of the previous sequence
        prev_suffix = prev_word_tokens[-k:]
        # The first k words of the new sequence
        new_prefix = new_word_tokens[:k]

        # To compare them fault-tolerantly, flatten the token IDs in the overlap
        flat_prev = [tok for word in prev_suffix for tok in word]
        flat_new = [tok for word in new_prefix for tok in word]

        # We can't guarantee the flattened lists are the same length
        # if the model tokenized slightly differently (e.g., "the" vs "thee").
        # So, we compare the shorter of the two.
        compare_len = min(len(flat_prev), len(flat_new))
        if compare_len == 0:
            continue

        matches = np.sum(
            np.array(flat_prev[:compare_len]) == np.array(flat_new[:compare_len])
        )

        # Scoring logic: ratio of matched tokens.
        # Add a small epsilon proportional to the number of words to favor longer word overlaps.
        matching_score = matches / compare_len + (k / 1000.0)

        if matching_score > max_matching_score:
            max_matching_score = matching_score
            best_overlap_n_words = k

    # If the best match is very poor, it's safer to not merge.
    # A threshold of 0.5 means at least half the tokens in the overlap should match.
    if max_matching_score < 0.5:
        best_overlap_n_words = 0

    # The final merged list is the previous list plus the non-overlapping part of the new one.
    merged = prev_words + new_words[best_overlap_n_words:]

    return merged


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> Tensor:
        """
        Perform forward pass through the preprocessor and encoder.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features, feature_lengths)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto
        the correct device and converting its format.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length

    def embed_audio(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
        )


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    @torch.inference_mode()
    def transcribe(self, wav_file: str) -> str:
        """
        Transcribes a short audio file into text.
        """
        raise NotImplementedError(
            'Not adapted to the new decoder, use "transcribe_longform_overlap" instead.'
        )
        wav, length = self.prepare_wav(wav_file)
        if length > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        return self.decoding.decode(self.head, encoded, encoded_len)[0]

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        return self.head(self.encoder(features, feature_lengths)[0])

    def to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx ASR model.
        `ctc`:  exported entirely in encoder-decoder format.
        `rnnt`: exported in encoder/decoder/joint parts separately.
        """
        if "ctc" in self.cfg.model_name:
            saved_forward = self.forward
            self.forward = self.forward_for_export
            onnx_converter(
                model_name=self.cfg.model_name,
                out_dir=dir_path,
                module=self,
                inputs=self.encoder.input_example(),
                input_names=["features", "feature_lengths"],
                output_names=["log_probs"],
                dynamic_axes={
                    "features": {0: "batch_size", 2: "seq_len"},
                    "feature_lengths": {0: "batch_size"},
                    "log_probs": {0: "batch_size", 1: "seq_len"},
                },
            )
            self.forward = saved_forward
        else:
            super().to_onnx(dir_path)  # export encoder
            onnx_converter(
                model_name=f"{self.cfg.model_name}_decoder",
                out_dir=dir_path,
                module=self.head.decoder,
            )
            onnx_converter(
                model_name=f"{self.cfg.model_name}_joint",
                out_dir=dir_path,
                module=self.head.joint,
            )

    @torch.inference_mode()
    def transcribe_longform(
        self, wav_file: str, **kwargs
    ) -> List[Dict[str, Union[str, Tuple[float, float]]]]:
        """
        Transcribes a long audio file by splitting it into segments and
        then transcribing each segment.
        """
        raise NotImplementedError(
            'Not adapted to the new decoder, use "transcribe_longform_overlap" instead.'
        )
        from .vad_utils import segment_audio

        transcribed_segments = []
        wav = load_audio(wav_file, return_format="int")
        segments, boundaries = segment_audio(
            wav, SAMPLE_RATE, device=self._device, **kwargs
        )
        for segment, segment_boundaries in zip(segments, boundaries):
            wav = segment.to(self._device).unsqueeze(0).to(self._dtype)
            length = torch.full([1], wav.shape[-1], device=self._device)
            encoded, encoded_len = self.forward(wav, length)
            result = self.decoding.decode(self.head, encoded, encoded_len)[0]
            transcribed_segments.append(
                {
                    "transcription": result,
                    "boundaries": segment_boundaries,
                }
            )
        return transcribed_segments

    ### --- CUSTOM IMPLEMENTATION --- ###

    @torch.inference_mode()
    def transcribe_longform_overlap(
        self,
        wav_file: str,
        chunk_len_sec: int = 20,
        overlap_len_sec: int = 4,
        sample_rate: int = 16000,
        batch_size: int = 8,  # <-- New parameter for batch size
        **kwargs,
    ):
        """
            Transcribes a long audio file using batched, overlapping chunks.
        """
        # 1. Chunking logic remains the same
        wav = load_audio(wav_file)
        chunk_samples = chunk_len_sec * sample_rate
        overlap_samples = overlap_len_sec * sample_rate
        step_samples = chunk_samples - overlap_samples

        chunks = []
        start = 0
        while start < len(wav):
            end = start + chunk_samples
            chunks.append(wav[start:end])
            start += step_samples

        # --- KEY CHANGE 1: Group chunks into batches ---
        chunk_batches = []
        for i in range(0, len(chunks), batch_size):
            chunk_batches.append(chunks[i:i + batch_size])
            
        print(f"Split audio into {len(chunks)} chunks, grouped into {len(chunk_batches)} batches of size up to {batch_size}.")

        final_word_timestamps = []
        global_chunk_index = 0  # Keep track of the absolute chunk number for timestamping

        # 2. Main loop now iterates over BATCHES of chunks
        for batch in chunk_batches:
            # --- KEY CHANGE 2: Pad chunks within the batch to the same length ---
            max_len_in_batch = max(len(c) for c in batch)
            
            padded_wavs = []
            original_lengths = []

            for chunk in batch:
                original_lengths.append(len(chunk))
                # Pad the numpy array before converting to a tensor
                padding_needed = max_len_in_batch - len(chunk)
                padded_chunk = np.pad(chunk, (0, padding_needed), mode='constant')
                padded_wavs.append(padded_chunk)

            # Create the batch tensors for the model
            wav_batch = torch.from_numpy(np.stack(padded_wavs)).to(self._device).to(self._dtype)
            length_batch = torch.tensor(original_lengths, device=self._device)

            # 3. Perform batched inference
            encoded, encoded_len = self.forward(wav_batch, length_batch)
            res = self.decoding.decode(self.head, encoded, encoded_len)
            
            # `res` contains results for the whole batch. 
            # `res["word_timestamps"]` is now a list of lists.
            batch_of_word_lists = res["word_timestamps"]
            # print(batch_of_word_lists)

            # --- KEY CHANGE 3: Process results of the batch sequentially ---
            # This inner loop handles the logic for each chunk from the batch result
            for i, new_words in enumerate(batch_of_word_lists):
                
                # Timestamp Correction: Use the global index
                current_chunk_idx = global_chunk_index + i
                time_offset_sec = (current_chunk_idx * step_samples) / sample_rate

                for word in new_words:
                    word["start"] = round(word["start"] + time_offset_sec, 2)
                    word["end"] = round(word["end"] + time_offset_sec, 2)

                # print(f"--- Chunk {current_chunk_idx + 1}: Found {len(new_words)} words (time offset: {time_offset_sec:.2f}s) ---")
                
                # Merging: The merge function is called sequentially for each result in the batch
                final_word_timestamps = _merge_word_timestamps(final_word_timestamps, new_words)
            
            # Update the global index to the start of the next batch
            global_chunk_index += len(batch)


        return final_word_timestamps


class GigaAMEmo(GigaAM):
    """
    Giga Acoustic Model for Emotion Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.id2name = cfg.id2name

    def get_probs(self, wav_file: str) -> Dict[str, float]:
        """
        Calculate probabilities for each emotion class based on the provided audio file.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, _ = self.forward(wav, length)
        encoded_pooled = nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1]
        ).squeeze(-1)

        logits = self.head(encoded_pooled)[0]
        probs = nn.functional.softmax(logits, dim=-1).detach().tolist()

        return {self.id2name[i]: probs[i] for i in range(len(self.id2name))}

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, _ = self.encoder(features, feature_lengths)
        enc_pooled = nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1].item()
        ).squeeze(-1)
        return nn.functional.softmax(self.head(enc_pooled)[0], dim=-1)

    def to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx Emo model.
        """
        saved_forward = self.forward
        self.forward = self.forward_for_export
        onnx_converter(
            model_name=self.cfg.model_name,
            out_dir=dir_path,
            module=self,
            inputs=self.encoder.input_example(),
            input_names=["features", "feature_lengths"],
            output_names=["probs"],
            dynamic_axes={
                "features": {0: "batch_size", 2: "seq_len"},
                "feature_lengths": {0: "batch_size"},
                "probs": {0: "batch_size", 1: "seq_len"},
            },
        )
        self.forward = saved_forward
