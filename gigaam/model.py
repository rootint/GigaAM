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

    # @torch.inference_mode()
    # def transcribe_longform_overlap(
    #     self,
    #     wav_file: str,
    #     chunk_len_sec: int = 20,
    #     overlap_len_sec: int = 4,
    #     sample_rate: int = 16000,
    #     **kwargs,
    # ):
    #     """
    #     Transcribes a long audio file by splitting it into overlapping chunks,
    #     then intelligently merging the resulting word timestamp lists.
    #     """
    #     # This part remains the same
    #     wav = load_audio(wav_file)
    #     chunk_samples = chunk_len_sec * sample_rate
    #     overlap_samples = overlap_len_sec * sample_rate
    #     step_samples = chunk_samples - overlap_samples

    #     chunks = []
    #     start = 0
    #     while start < len(wav):
    #         end = start + chunk_samples
    #         chunks.append(wav[start:end])
    #         start += step_samples

    #     # CHANGE: We will store the final list of word dictionaries here.
    #     final_word_timestamps = []

    #     for i, chunk in enumerate(chunks):
    #         # Audio processing remains the same
    #         wav_chunk = chunk.to(self._device).to(self._dtype).unsqueeze(0)
    #         length = torch.full([1], wav_chunk.shape[-1], device=self._device)
    #         encoded, encoded_len = self.forward(wav_chunk, length)

    #         # Get the structured result from the decoder
    #         res = self.decoding.decode(self.head, encoded, encoded_len)

    #         # We work with the word_timestamps list now
    #         new_words = res["word_timestamps"]  # Assuming decode returns a batch

    #         # --- KEY CHANGE 1: Timestamp Correction ---
    #         # Calculate the time offset of the current chunk in seconds.
    #         time_offset_sec = (i * step_samples) / sample_rate

    #         # Apply the offset to make timestamps absolute.
    #         for word in new_words:
    #             word["start"] += time_offset_sec
    #             word["end"] += time_offset_sec
    #             # Round for cleaner output, optional
    #             word["start"] = round(word["start"], 2)
    #             word["end"] = round(word["end"], 2)

    #         print(
    #             f"--- Chunk {i+1}: Found {len(new_words)} words (time offset: {time_offset_sec:.2f}s) ---"
    #         )
    #         # For debugging, you can print the words from this chunk:
    #         # print([w['word'] for w in new_words])

    #         # --- KEY CHANGE 2: Merging Word Lists ---
    #         # Use our new helper function to merge the word lists.
    #         final_word_timestamps = _merge_word_timestamps(
    #             final_word_timestamps, new_words
    #         )

    #     # The final result is the merged list of word dictionaries.
    #     # You can return this directly or format it as a string.
    #     return final_word_timestamps

    def transcribe_longform_overlap(
        self,
        wav_file: str,
        chunk_len_sec: int = 20,
        overlap_len_sec: int = 4,
        sample_rate: int = 16000,
        **kwargs,
    ):
        """
        Implements the full long-form transcription pipeline using a
        fault-tolerant merging strategy inspired by Hugging Face's implementation.
        """
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

        final_sequence = []

        for i, chunk in enumerate(chunks):
            wav_chunk = chunk.to(self._device).to(self._dtype).unsqueeze(0)
            length = torch.full([1], wav_chunk.shape[-1], device=self._device)
            encoded, encoded_len = self.forward(wav_chunk, length)

            # Get the structured result from the decoder
            res = self.decoding.decode(self.head, encoded, encoded_len)

            new_sequence = res['ids'][0]
            new_words = res['word_time']

            # 5) Merge the tokens using the fault-tolerant algorithm
            if not final_sequence:
                final_sequence = new_sequence
            else:
                # This block implements the Hugging Face merge logic
                best_overlap_index = 0
                max_matching_score = 0.0

                # We check every possible overlap length `k`
                for k in range(1, min(len(final_sequence), len(new_sequence)) + 1):
                    # Suffix of the previous sequence
                    prev_suffix = final_sequence[-k:]
                    # Prefix of the new sequence
                    new_prefix = new_sequence[:k]

                    # Count how many tokens match
                    matches = np.sum(np.array(prev_suffix) == np.array(new_prefix))

                    # Calculate the matching score with an epsilon for tie-breaking
                    epsilon = k / 10000.0
                    matching_score = matches / k + epsilon

                    # If this is the best score so far, store it
                    # We add `matches > 1` as a heuristic to avoid spurious single-token matches
                    if matches > 1 and matching_score > max_matching_score:
                        max_matching_score = matching_score
                        best_overlap_index = k

                if best_overlap_index > 0:
                    # Append the part of the new tokens that comes *after* the overlap
                    final_sequence.extend(new_sequence[best_overlap_index:])
                else:
                    # print("No significant overlap found. Appending all new tokens.")
                    final_sequence.extend(new_sequence)

        return self.decoding.tokenizer.decode(final_sequence)


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
