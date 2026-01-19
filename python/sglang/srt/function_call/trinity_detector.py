"""
Detector for Trinity models using Qwen-style function call format.

Handles tool calls inside <think> sections by stripping the tags before parsing.
"""

import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class TrinityDetector(BaseFormatDetector):
    """
    Detector for Trinity models using Qwen-style function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>
    ```

    This detector handles tool calls that appear inside <think> sections
    by stripping the think tags before parsing.

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>\n"
        self.eot_token = "\n</tool_call>"
        self.tool_call_separator = "\n"
        self._normal_text_buffer = ""

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think> and </think> tags, keeping the content inside."""
        return text.replace("<think>", "").replace("</think>", "")

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a tool call."""
        text = self._strip_think_tags(text)
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: StreamingParseResult with normal text and parsed calls.
        """
        text = self._strip_think_tags(text)

        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            try:
                parsed_call = json.loads(match_result.strip())
                calls.extend(self.parse_base_json(parsed_call, tools))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}"
                )
                continue
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for tool calls.
        Uses base class implementation with buffering to handle partial end tokens.
        """
        new_text = self._strip_think_tags(new_text)
        result = super().parse_streaming_increment(new_text, tools)

        if result.normal_text:
            self._normal_text_buffer += result.normal_text

            end_token_without_newline = self.eot_token[1:]  # "</tool_call>"
            if end_token_without_newline in self._normal_text_buffer:
                cleaned_text = self._normal_text_buffer.replace(
                    end_token_without_newline, ""
                )
                self._normal_text_buffer = ""
                result.normal_text = cleaned_text
            else:
                partial_match_len = self._ends_with_partial_token(
                    self._normal_text_buffer, end_token_without_newline
                )

                if partial_match_len:
                    result.normal_text = self._normal_text_buffer[:-partial_match_len]
                    self._normal_text_buffer = self._normal_text_buffer[
                        -partial_match_len:
                    ]
                else:
                    result.normal_text = self._normal_text_buffer
                    self._normal_text_buffer = ""

        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n{"name":"' + name + '", "arguments":',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )
