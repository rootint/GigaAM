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

        # for i in range(b):
        #     pred_texts.append(
        #         "".join(self.tokenizer.decode(labels[i][skip_mask[i]].cpu().tolist()))
        #     )
        # return pred_texts
        # print('skip mask', skip_mask[0])
        # pred_texts: List[str] = []
        pred_ids: List[int] = []
        word_timestamps = []
        current_word_tokens = []
        word_begin_ms = -1
        word_end_ms = 0
        lbls = labels.cpu().tolist()
        for i in range(b):
            for idx, is_masked in enumerate(skip_mask[i]):
                if is_masked and lbls[i][idx] != 0:
                    if word_begin_ms == -1:
                        word_begin_ms = idx * 40
                    current_word_tokens.append(lbls[i][idx])
                    # print(lbls[i][idx], self.tokenizer.decode([lbls[i][idx]]))
                    word_end_ms = idx * 40
                elif is_masked and lbls[i][idx] == 0:
                    current_word_tokens.append(lbls[i][idx])
                    word_timestamps.append(
                        {
                            "word": self.tokenizer.decode(current_word_tokens),
                            "token_ids": current_word_tokens,
                            "start": word_begin_ms / 1000,
                            "end": word_end_ms / 1000,
                        }
                    )
                    current_word_tokens = []
                    word_begin_ms = -1
            pred_ids.append(labels[i][skip_mask[i]].cpu().tolist())
            # pred_texts.append(
            #     "".join(self.tokenizer.decode(labels[i][skip_mask[i]].cpu().tolist()))
            # )
        if current_word_tokens:
            word_timestamps.append(
                {
                    "word": self.tokenizer.decode(current_word_tokens),
                    "token_ids": current_word_tokens,
                    "start": word_begin_ms / 1000,
                    "end": word_end_ms / 1000,
                }
            )
        # print(pred_ids[0])
        # print([token_id for i in word_timestamps for token_id in (i["token_ids"])])
        # print(
        #     "checking",
        #     pred_ids[0]
        #     == [token_id for i in word_timestamps for token_id in (i["token_ids"])],
        # )
        return {
            # "text": pred_texts,
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
