"""XingChen4-specific incremental detokenization compatibility hook."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TARGET = (
    "sglang.srt.managers.detokenizer_manager."
    "DetokenizerManager._decode_batch_token_id_output"
)


def _is_xingchen4_tokenizer(tokenizer) -> bool:
    current = tokenizer
    for _ in range(3):
        cls = type(current)
        class_and_module = (cls.__name__, cls.__module__)
        if class_and_module[0] == "XingChen4Tokenizer" and "tokenization_xingchen4" in class_and_module[1]:
            return True
        if class_and_module[0] == "Xing4_0Tokenizer" and "tokenization_xing4_0" in class_and_module[1]:
            return True
        name_or_path = str(getattr(current, "name_or_path", ""))
        if any(
            model_name in name_or_path
            for model_name in ("XingChen4-29B-A4B-0907", "Xing4.0-29B-A4B")
        ):
            return True
        inner = getattr(current, "tokenizer", None)
        if inner is None or inner is current:
            break
        current = inner
    return False


def _decode_xingchen4(self, recv_obj):
    """SGLang incremental decode with byte-fallback recovery at any position."""
    from sglang.srt.managers.detokenizer_manager import DecodeStatus

    bs = len(recv_obj.rids)
    vocab_size = self.vocab_size

    read_ids, surr_ids = [], []
    for i in range(bs):
        rid = recv_obj.rids[i]
        if rid not in self.decode_status:
            status = DecodeStatus(
                decoded_text=recv_obj.decoded_texts[i],
                decode_ids=self._clamp_decode_ids(recv_obj.decode_ids[i], vocab_size),
                surr_offset=0,
                read_offset=recv_obj.read_offsets[i],
            )
            self.decode_status[rid] = status
        else:
            status = self.decode_status[rid]
            status.decode_ids.extend(
                self._clamp_decode_ids(recv_obj.decode_ids[i], vocab_size)
            )

        read_ids.append(
            self.trim_matched_stop(
                status.decode_ids[status.surr_offset :],
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
        )
        surr_ids.append(
            status.decode_ids[status.surr_offset : status.read_offset]
        )

    if not self.disable_tokenizer_batch_decode:
        surr_texts = self._grouped_batch_decode(
            surr_ids,
            recv_obj.skip_special_tokens,
            recv_obj.spaces_between_special_tokens,
        )
        read_texts = self._grouped_batch_decode(
            read_ids,
            recv_obj.skip_special_tokens,
            recv_obj.spaces_between_special_tokens,
        )
    else:
        surr_texts = [
            self.tokenizer.decode(
                ids,
                skip_special_tokens=skip,
                spaces_between_special_tokens=spaces,
            )
            for ids, skip, spaces in zip(
                surr_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
        ]
        read_texts = [
            self.tokenizer.decode(
                ids,
                skip_special_tokens=skip,
                spaces_between_special_tokens=spaces,
            )
            for ids, skip, spaces in zip(
                read_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
        ]

    output_strs = []
    for i in range(bs):
        rid = recv_obj.rids[i]
        try:
            status = self.decode_status[rid]
        except KeyError as exc:
            raise RuntimeError(
                f"Decode status not found for request {rid}. It may have been "
                "evicted because of memory pressure."
            ) from exc

        new_text = read_texts[i][len(surr_texts[i]) :]
        if recv_obj.finished_reasons[i] is None:
            pending = status.sent_offset - status.decoded_text_len
            if new_text and "�" not in new_text:
                status.append_decoded_text(new_text)
                status.surr_offset = status.read_offset
                status.read_offset = len(status.decode_ids)
                status.sent_offset = status.decoded_text_len
                output_strs.append(new_text[pending:] if pending else new_text)
            else:
                # SentencePiece byte fallback can place U+FFFD before already
                # printable ASCII in the same chunk (for example, "�infty").
                # The generic helper only withholds a trailing U+FFFD, so keep
                # everything from the first replacement character pending.
                printable = new_text.split("�", 1)[0]
                status.sent_offset = status.decoded_text_len + len(printable)
                output_strs.append(printable[pending:] if pending else printable)
            continue

        del self.decode_status[rid]
        accumulated_text = status.get_decoded_text() + new_text
        output_str = self.trim_matched_stop(
            accumulated_text,
            recv_obj.finished_reasons[i],
            recv_obj.no_stop_trim[i],
        )
        if "�" in output_str:
            # Incremental SentencePiece decoding can split byte-fallback token
            # sequences at a window boundary.  Re-decode the complete output-id
            # sequence once at finish; this recovers valid UTF-8 instead of
            # deleting characters or truncating everything after the marker.
            final_ids = self.trim_matched_stop(
                status.decode_ids,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
            full_text = self.tokenizer.decode(
                final_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[i],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[i],
            )
            full_replacements = full_text.count("�")
            old_replacements = output_str.count("�")
            if full_replacements < old_replacements:
                logger.warning(
                    "Recovered Xing4.0 final output by full decode for request %s: U+FFFD %d -> %d",
                    rid,
                    old_replacements,
                    full_replacements,
                )
                output_str = full_text
        incremental_output = output_str[status.sent_offset :]
        status.sent_offset = len(output_str)
        output_strs.append(incremental_output)

    for i, output_str in enumerate(output_strs):
        if "�" in output_str:
            logger.warning(
                "Withholding U+FFFD at XingChen4 detokenizer output boundary for request %s",
                recv_obj.rids[i],
            )
            output_strs[i] = output_str.split("�", 1)[0]
    return output_strs


def _around_decode(original_fn, self, recv_obj):
    if not _is_xingchen4_tokenizer(self.tokenizer):
        return original_fn(self, recv_obj)
    return _decode_xingchen4(self, recv_obj)


def register_xingchen4_detokenizer_hook() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(_TARGET, _around_decode, HookType.AROUND)
    logger.info("XingChen4 byte-fallback detokenizer hook registered")
