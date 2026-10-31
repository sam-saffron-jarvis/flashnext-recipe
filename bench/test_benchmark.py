"""Pure CPU tests; no socket, network, GPU or engine calls."""
import argparse
import io
import json
import unittest
from unittest.mock import patch
import benchmark


class Response(io.BytesIO):
    pass


class BenchmarkTests(unittest.TestCase):
    def test_sse_ignores_comments_and_done(self):
        data = b': ping\n\ndata: {"a":1}\n\ndata: [DONE]\n\n'
        self.assertEqual(list(benchmark.events(io.BytesIO(data))), [{'a': 1}])

    def test_prompt_unique_and_target(self):
        a, n = benchmark.facts(54000, 'READY')
        b, m = benchmark.facts(54000, 'READY')
        self.assertEqual((n, m), (1589, 1589))
        self.assertNotEqual(a, b)
        self.assertIn('Fact 1588:', a)

    def test_streamed_tools_and_timing_fields(self):
        values = [
            {'choices': [{'delta': {'role': 'assistant', 'content': ''}}]},
            {'choices': [{'delta': {'reasoning_content': 'brief'}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function', 'function': {'name': 'echo', 'arguments': ''}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"text":"READY"}'}}]}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}],
             'usage': {'prompt_tokens': 2000, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 3}},
             'timings': {'cache_n': 3, 'prompt_n': 1997, 'predicted_per_second': 70}}
        ]
        wire = ''.join('data: ' + json.dumps(v) + '\n\n' for v in values) + 'data: [DONE]\n\n'
        args = argparse.Namespace(model='strata', url='http://unused.test/v1/chat/completions', timeout=1)
        with patch('urllib.request.urlopen', return_value=Response(wire.encode())):
            row = benchmark.call(args, {'max_tokens': 1000}, 'mock')
        self.assertEqual(row['tool_calls'], [{'id': 'call_1', 'name': 'echo', 'arguments': '{"text":"READY"}'}])
        self.assertEqual(row['reasoning_chars'], 5)
        self.assertTrue(row['early_stop'])
        self.assertTrue(row['prompt_reuse_detected'])
        self.assertEqual(row['timings']['predicted_per_second'], 70)
        self.assertEqual(row['finish_reason'], 'tool_calls')


if __name__ == '__main__':
    unittest.main()
