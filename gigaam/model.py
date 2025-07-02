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
    #     Implements the full long-form transcription pipeline using a
    #     fault-tolerant merging strategy inspired by Hugging Face's implementation.
    #     """
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

    #     final_sequence = []

    #     for i, chunk in enumerate(chunks):
    #         wav_chunk = chunk.to(self._device).to(self._dtype).unsqueeze(0)
    #         length = torch.full([1], wav_chunk.shape[-1], device=self._device)
    #         encoded, encoded_len = self.forward(wav_chunk, length)

    #         # Get the structured result from the decoder
    #         res = self.decoding.decode(self.head, encoded, encoded_len)

    #         new_sequence = res['ids'][0]
    #         new_words = res['word_timestamps']

    #         # 5) Merge the tokens using the fault-tolerant algorithm
    #         if not final_sequence:
    #             final_sequence = new_sequence
    #         else:
    #             # This block implements the Hugging Face merge logic
    #             best_overlap_index = 0
    #             max_matching_score = 0.0

    #             # We check every possible overlap length `k`
    #             for k in range(1, min(len(final_sequence), len(new_sequence)) + 1):
    #                 # Suffix of the previous sequence
    #                 prev_suffix = final_sequence[-k:]
    #                 # Prefix of the new sequence
    #                 new_prefix = new_sequence[:k]

    #                 # Count how many tokens match
    #                 matches = np.sum(np.array(prev_suffix) == np.array(new_prefix))

    #                 # Calculate the matching score with an epsilon for tie-breaking
    #                 epsilon = k / 10000.0
    #                 matching_score = matches / k + epsilon

    #                 # If this is the best score so far, store it
    #                 # We add `matches > 1` as a heuristic to avoid spurious single-token matches
    #                 if matches > 1 and matching_score > max_matching_score:
    #                     max_matching_score = matching_score
    #                     best_overlap_index = k

    #             if best_overlap_index > 0:
    #                 # Append the part of the new tokens that comes *after* the overlap
    #                 final_sequence.extend(new_sequence[best_overlap_index:])
    #             else:
    #                 # print("No significant overlap found. Appending all new tokens.")
    #                 final_sequence.extend(new_sequence)

    #     return self.decoding.tokenizer.decode(final_sequence)

    @torch.inference_mode()
    def transcribe_longform_overlap(
        self,
        wav_file: str,
        chunk_len_sec: int = 20,
        stride_len_sec: int = 16,
        sample_rate: int = 16000,
        batch_size: int = 8,  # New parameter for batching
        **kwargs,
    ):
        """
        Implements the full long-form transcription pipeline using a
        fault-tolerant merging strategy with batched inference.

        Args:
            wav_file (str): Path to the audio file.
            chunk_len_sec (int): Length of each audio chunk in seconds.
            overlap_len_sec (int): Length of the overlap between consecutive chunks in seconds.
            sample_rate (int): The sample rate of the audio.
            batch_size (int): Number of chunks to process simultaneously.
            **kwargs: Additional arguments.
        """
        wav = load_audio(wav_file)
        chunk_samples = chunk_len_sec * sample_rate
        stride_samples = stride_len_sec * sample_rate

        # 1) Split audio into overlapping chunks
        chunks = []
        start = 0
        while start < len(wav):
            end = start + chunk_samples
            chunks.append(wav[start:end])
            start += stride_samples

        final_sequence = []
        final_word_sequence = []
        final_words = []
        end = ""
        word_t_offset = 0

        # 2) Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_of_chunks = chunks[i : i + batch_size]

            # --- Create batch tensors for the model ---
            original_lengths = [len(c) for c in batch_of_chunks]
            max_len = max(original_lengths)

            # Pad all chunks in the batch to the same length
            padded_chunks = [
                torch.nn.functional.pad(c, (0, max_len - len(c)))
                for c in batch_of_chunks
            ]

            wav_batch = torch.stack(padded_chunks).to(self._device).to(self._dtype)
            length_batch = torch.tensor(original_lengths, device=self._device)

            # --- Perform batched inference ---
            encoded, encoded_len = self.forward(wav_batch, length_batch)
            res = self.decoding.decode(self.head, encoded, encoded_len)

            # --- De-batch results and stitch them sequentially ---
            # The `res` dictionary contains lists of results for each item in the batch
            batch_sequences = res["ids"]
            batch_word_timestamps = res["word_timestamps"]
            # for item in batch_sequences:
            #     print("begin", self.decoding.tokenizer.decode(item["begin_strided"]))
            #     print("non", self.decoding.tokenizer.decode(item["mid_strided"]))
            #     print("end", self.decoding.tokenizer.decode(item["end_strided"]))
            #     print()
            #     end = self.decoding.tokenizer.decode(item["end_strided"])

            #     if not final_sequence:
            #         final_sequence += self.decoding.tokenizer.decode(
            #             item["begin_strided"]
            #         ) + self.decoding.tokenizer.decode(item["mid_strided"])
            #     else:
            #         final_sequence += self.decoding.tokenizer.decode(
            #             item["mid_strided"]
            #         )

            for item in batch_word_timestamps:
                # print(
                #     "begin wt",
                #     "".join([word["word"] for word in item["begin_strided"]]),
                # )
                # print("mid wt", "".join([word["word"] for word in item["mid_strided"]]))
                # print("end wt", "".join([word["word"] for word in item["end_strided"]]))
                # print()
                stitched_last_word = False
                if not final_words:
                    final_words.extend(item["begin_strided"])
                    final_words.extend(item["mid_strided"])
                else:
                    if len(item['mid_strided']) == 0:
                        word_t_offset += stride_len_sec
                        continue
                    if (
                        final_words[-1]["word"][-1] != " "
                        and item["mid_strided"][0]["word"][0] != " "
                    ):
                        # print(final_words[-1]["word"], item["mid_strided"][0]["word"])
                        final_words[-1] = {
                            "word": final_words[-1]["word"]
                            + item["mid_strided"][0]["word"],
                            "token_ids": final_words[-1]["token_ids"]
                            + item["mid_strided"][0]["token_ids"],
                            "start": final_words[-1]["start"] + word_t_offset,
                            "end": item["mid_strided"][0]["end"] + word_t_offset,
                        }
                        stitched_last_word = True
                    for word in item["mid_strided"]:
                        if stitched_last_word:
                            stitched_last_word = False
                            continue
                        final_words.append(
                            {
                                "word": word["word"],
                                "token_ids": word["token_ids"],
                                "start": word["start"] + word_t_offset,
                                "end": word["end"] + word_t_offset,
                            }
                        )
                        # print(final_words)
                    # final_words.extend(item["mid_strided"])

                word_t_offset += stride_len_sec
                # print()

        for word in batch_word_timestamps[-1]["end_strided"]:
            final_words.append(
                {
                    "word": word["word"],
                    "token_ids": word["token_ids"],
                    "start": word["start"] + word_t_offset,
                    "end": word["end"] + word_t_offset,
                }
            )

        # final_sequence += end
        # return "".join(final_sequence)
        return final_words


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
