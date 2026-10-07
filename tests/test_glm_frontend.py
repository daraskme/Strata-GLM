import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "serve")]
from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer
from frontend import OutputParser, parse_tool_call


class GlmFrontendTest(unittest.TestCase):
    def test_pack_preserves_glm_whole_vocab_and_special_ids(self):
        # An exact GLM vocabulary hit is used even without a BPE merge rule.
        # The previous API loader selected Qwen and incorrectly emitted 4 IDs.
        tokens = list(BYTE_TO_UNICODE.values()) + ["test", "<|user|>"]
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as td:
            path = Path(td)
            (path / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}))
            (path / "token_type.json").write_text(json.dumps([1]*257 + [3]))
            (path / "merges.txt").write_text("")
            cfg = {"model": "gpt2", "pre": "glm4", "special_ids": {"tokenizer.ggml.eos_token_id": 257}}
            (path / "tokenizer.json").write_text(json.dumps(cfg))
            tok = Tokenizer.from_pack(path)
            self.assertEqual(tok.encode("test"), [256])
            self.assertTrue(tok.ignore_merges)
            self.assertEqual(tok.special_ids, cfg["special_ids"])
            for s in ["日本語の会話。", "1234567\n\n", "e\u0301 🐈"]:
                self.assertEqual(tok.decode(tok.encode(s)), s)
            cfg["pre"] = "unknown"
            (path / "tokenizer.json").write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                Tokenizer.from_pack(path)

    def test_glm_tools_all_chunk_boundaries(self):
        tools = [{"name": "write_file", "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "count": {"type": "integer"}, "items": {"type": "array"}}}}]
        body = ('write_file<arg_key>text</arg_key><arg_value>007\n日本語</arg_value>'
                '<arg_key>count</arg_key><arg_value>2</arg_value>'
                '<arg_key>items</arg_key><arg_value>["a", "b"]</arg_value>')
        s = "考えました。</think><tool_call>" + body + "</tool_call>\n<tool_call>ping</tool_call>"
        for n in [1, 2, 7, len(s)]:
            parser = OutputParser(thinking=True, tools=tools, stream_tools=True)
            events = []
            for i in range(0, len(s), n):
                events += parser.feed(s[i:i+n])
            events += parser.finish()
            calls = [e.call for e in events if e.kind == "tool_call"]
            self.assertEqual([c.name for c in calls], ["write_file", "ping"])
            self.assertEqual(calls[0].arguments, {"text": "007\n日本語", "count": 2, "items": ["a", "b"]})
            self.assertEqual(calls[1].arguments, {})
            self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), "考えました。")
            self.assertFalse(any(e.kind == "content" and e.text.strip() for e in events))

    def test_qwen_tools_keep_streaming(self):
        parser = OutputParser(thinking=False, stream_tools=True)
        s = '<tool_call><function=lookup><parameter=n>3</parameter></function></tool_call>'
        events = []
        for ch in s:
            events += parser.feed(ch)
        call = next(e.call for e in events if e.kind == "tool_call")
        self.assertEqual(call.name, "lookup")
        self.assertEqual(call.arguments, {"n": 3})
        self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), {"n": 3})

    def test_malformed_glm_is_rejected(self):
        for body in ["", "f<arg_key>x</arg_key>", "f<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value>"]:
            with self.assertRaises(ValueError):
                parse_tool_call(body)


if __name__ == "__main__":
    unittest.main()
