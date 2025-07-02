from typing import List, Optional

import torch
from sentencepiece import SentencePieceProcessor
from torch import Tensor

from .decoder import CTCHead, RNNTHead


class Tokenizer:
    """
    Tokenizer for converting between text and token IDs.
    The tokenizer can operate either character-wise or using a pre-trained SentencePiece model.
    """

    def __init__(self, vocab: List[str], model_path: Optional[str] = None):
        self.charwise = model_path is None
        if self.charwise:
            self.vocab = vocab
        else:
            self.model = SentencePieceProcessor()
            self.model.load(model_path)

    def decode(self, tokens: List[int]) -> str:
        """
        Convert a list of token IDs back to a string.
        """
        if self.charwise:
            return "".join(self.vocab[tok] for tok in tokens)
        return self.model.decode(tokens)

    def __len__(self):
        """
        Get the total number of tokens in the vocabulary.
        """
        return len(self.vocab) if self.charwise else len(self.model)


class CTCGreedyDecoding:
    """
    Class for performing greedy decoding of CTC outputs.
    """

    def __init__(self, vocabulary: List[str], model_path: Optional[str] = None):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)

    @torch.inference_mode()
    def decode(self, head: CTCHead, encoded: Tensor, lengths: Tensor) -> List[str]:
        """
        Decode the output of a CTC model into a list of hypotheses.
        """
        log_probs = head(encoder_output=encoded)

        STRIDE_N = int(4 * 1000 / 40 / 2)
        assert (
            len(log_probs.shape) == 3
        ), f"Expected log_probs shape {log_probs.shape} == [B, T, C]"
        b, _, c = log_probs.shape
        assert (
            c == len(self.tokenizer) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.tokenizer) + 1}"
        labels = log_probs.argmax(dim=-1, keepdim=False)
        # print("len labels", len(labels[0]), b)
        # # print('labels', labels)
        # lbls = labels[0].cpu().tolist()
        # text = ""
        # for i in lbls:
        #     if i == 33:
        #         text += "Z"
        #     else:
        #         text += self.tokenizer.decode([i])
        # print("labels", text)
        # print("\n\n")
        skip_mask = labels != self.blank_id
        skip_mask[:, 1:] = torch.logical_and(
            skip_mask[:, 1:], labels[:, 1:] != labels[:, :-1]
        )
        for length in lengths:
            skip_mask[length:] = 0

        pred_ids: List[dict] = []
        word_timestamps = []

        for i in range(b):
            begin_strided_ids: List[int] = []
            mid_strided_ids: List[int] = []
            end_strided_ids: List[int] = []
            begin_strided_wt: List[dict] = []
            mid_strided_wt: List[dict] = []
            end_strided_wt: List[dict] = []
            current_word_tokens = []
            word_begin_ms = None
            word_end_ms = None
            current_labels = labels[i].cpu().tolist()
            for idx, is_masked in enumerate(skip_mask[i]):
                if is_masked and idx <= STRIDE_N:
                    begin_strided_ids.append(current_labels[idx])
                    current_word_tokens.append(current_labels[idx])

                    # if space
                    if current_labels[idx] == 0:
                        word_end_ms = idx * 40
                        begin_strided_wt.append(
                            {
                                "word": self.tokenizer.decode(current_word_tokens),
                                "token_ids": current_word_tokens,
                                "start": word_begin_ms / 1000 if word_begin_ms else idx * 40,
                                "end": word_end_ms / 1000,
                            }
                        )
                        word_begin_ms = None
                        current_word_tokens = []
                    else:
                        if not word_begin_ms:
                            word_begin_ms = idx * 40

                elif is_masked and idx >= len(skip_mask[i]) - (STRIDE_N):
                    if len(current_word_tokens) > 0 and not end_strided_ids:
                        word_end_ms = idx * 40
                        mid_strided_wt.append(
                            {
                                "word": self.tokenizer.decode(current_word_tokens),
                                "token_ids": current_word_tokens,
                                "start": word_begin_ms / 1000 if word_begin_ms else idx * 40,
                                "end": word_end_ms / 1000,
                            }
                        )
                        word_begin_ms = None
                        current_word_tokens = []

                    end_strided_ids.append(current_labels[idx])
                    current_word_tokens.append(current_labels[idx])
                    # if space
                    if current_labels[idx] == 0:
                        word_end_ms = idx * 40
                        end_strided_wt.append(
                            {
                                "word": self.tokenizer.decode(current_word_tokens),
                                "token_ids": current_word_tokens,
                                "start": word_begin_ms / 1000 if word_begin_ms else idx * 40,
                                "end": word_end_ms / 1000,
                            }
                        )
                        word_begin_ms = None
                        current_word_tokens = []
                    else:
                        if not word_begin_ms:
                            word_begin_ms = idx * 40
                elif is_masked:
                    if len(current_word_tokens) > 0 and not mid_strided_ids:
                        word_end_ms = idx * 40
                        begin_strided_wt.append(
                            {
                                "word": self.tokenizer.decode(current_word_tokens),
                                "token_ids": current_word_tokens,
                                "start": word_begin_ms / 1000 if word_begin_ms else idx * 40,
                                "end": word_end_ms / 1000,
                            }
                        )
                        word_begin_ms = None
                        current_word_tokens = []
                        
                    current_word_tokens.append(current_labels[idx])
                    mid_strided_ids.append(current_labels[idx])

                    # if space or end of stride
                    if current_labels[idx] == 0:
                        word_end_ms = idx * 40
                        mid_strided_wt.append(
                            {
                                "word": self.tokenizer.decode(current_word_tokens),
                                "token_ids": current_word_tokens,
                                "start": word_begin_ms / 1000 if word_begin_ms else idx * 40,
                                "end": word_end_ms / 1000,
                            }
                        )
                        word_begin_ms = None
                        current_word_tokens = []
                    else:
                        if not word_begin_ms:
                            word_begin_ms = idx * 40
            if current_word_tokens:
                word_end_ms = idx * 40
                end_strided_wt.append(
                    {
                        "word": self.tokenizer.decode(current_word_tokens),
                        "token_ids": current_word_tokens,
                        "start": word_begin_ms / 1000 if word_begin_ms else idx * 40,
                        "end": word_end_ms / 1000,
                    }
                )
                word_begin_ms = None
                current_word_tokens = []
            pred_ids.append(
                {
                    "mid_strided": mid_strided_ids,
                    "begin_strided": begin_strided_ids,
                    "end_strided": end_strided_ids,
                }
            )

            word_timestamps.append(
                {
                    "mid_strided": mid_strided_wt,
                    "begin_strided": begin_strided_wt,
                    "end_strided": end_strided_wt,
                }
            )

        return {
            "ids": pred_ids,
            "word_timestamps": word_timestamps,
        }


class RNNTGreedyDecoding:
    def __init__(
        self,
        vocabulary: List[str],
        model_path: Optional[str] = None,
        max_symbols_per_step: int = 10,
    ):
        """
        Class for performing greedy decoding of RNN-T outputs.
        """
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.max_symbols = max_symbols_per_step

    def _greedy_decode(self, head: RNNTHead, x: Tensor, seqlen: Tensor) -> str:
        """
        Internal helper function for performing greedy decoding on a single sequence.
        """
        hyp: List[int] = []
        dec_state: Optional[Tensor] = None
        last_label: Optional[Tensor] = None
        for t in range(seqlen):
            f = x[t, :, :].unsqueeze(1)
            not_blank = True
            new_symbols = 0
            while not_blank and new_symbols < self.max_symbols:
                g, hidden = head.decoder.predict(last_label, dec_state)
                k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
                if k == self.blank_id:
                    not_blank = False
                else:
                    hyp.append(k)
                    dec_state = hidden
                    last_label = torch.tensor([[hyp[-1]]]).to(x.device)
                    new_symbols += 1

        return self.tokenizer.decode(hyp)

    @torch.inference_mode()
    def decode(self, head: RNNTHead, encoded: Tensor, enc_len: Tensor) -> List[str]:
        """
        Decode the output of an RNN-T model into a list of hypotheses.
        """
        b = encoded.shape[0]
        pred_texts = []
        encoded = encoded.transpose(1, 2)
        for i in range(b):
            inseq = encoded[i, :, :].unsqueeze(1)
            pred_texts.append(self._greedy_decode(head, inseq, enc_len[i]))
        return pred_texts
