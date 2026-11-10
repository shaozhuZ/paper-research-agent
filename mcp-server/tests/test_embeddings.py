"""Provider integration checks without API calls or Milvus writes."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embeddings import EmbeddingConfig, VoyageEmbeddings, validate_collection


def reply(texts=1, dimension=256, status=200):
    response = Mock(ok=status == 200, status_code=status, headers={})
    response.json.return_value = {
        "model": "voyage-4", "usage": {"total_tokens": 10},
        "data": [{"index": i, "embedding": [float(i + 1)] * dimension} for i in reversed(range(texts))],
    }
    return response


class TestVoyageEmbeddings(unittest.TestCase):
    def make(self):
        return VoyageEmbeddings(dimension=256, api_key="test-only-key")

    @patch("embeddings.requests.post")
    def test_document_query_roles_and_returned_order(self, post):
        post.side_effect = [reply(2), reply()]
        model = self.make()
        vectors = model.embed_documents(["first", "second"])
        self.assertEqual([v[0] for v in vectors], [1.0, 2.0])
        model.embed_query("question")
        document, query = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual(document["input_type"], "document")
        self.assertEqual(query["input_type"], "query")
        self.assertFalse(document["truncation"])
        self.assertEqual(document["output_dtype"], "float")
        self.assertEqual(model.total_tokens, 20)

    @patch("embeddings.requests.post", return_value=reply(status=403))
    @patch("embeddings.time.sleep")
    def test_permission_failure_is_not_retried_or_exposes_key(self, sleep, post):
        with self.assertRaises(RuntimeError) as error:
            self.make().embed_query("question")
        post.assert_called_once()
        sleep.assert_not_called()
        self.assertNotIn("test-only-key", str(error.exception))

    @patch("embeddings.time.sleep")
    @patch("embeddings.requests.post")
    def test_transient_rate_limit_obeys_bounded_retry(self, post, sleep):
        limited = reply(status=429)
        limited.headers = {"Retry-After": "120"}
        post.side_effect = [limited, reply()]
        self.make().embed_query("question")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(30)

    @patch("embeddings.requests.post")
    def test_wrong_dimension_and_duplicate_indices_are_rejected(self, post):
        post.return_value = reply(dimension=512)
        with self.assertRaisesRegex(RuntimeError, "invalid embedding"):
            self.make().embed_query("question")
        duplicate = reply(2)
        duplicate.json.return_value["data"][1]["index"] = 1
        post.return_value = duplicate
        with self.assertRaisesRegex(RuntimeError, "indices/count"):
            self.make().embed_documents(["first", "second"])


class TestCollectionIdentity(unittest.TestCase):
    def description(self, metadata=None, dimension=1024):
        return {"fields": [{"name": "vector", "params": {"dim": dimension}}],
                "description": "embedding:" + json.dumps(metadata) if metadata else ""}

    def test_same_dimension_different_model_is_rejected(self):
        config = EmbeddingConfig("voyage", "voyage-4", 1024)
        validate_collection(config, self.description(config.metadata(1024)))
        wrong = EmbeddingConfig("voyage", "voyage-3.5", 1024)
        with self.assertRaisesRegex(ValueError, "provider/model"):
            validate_collection(config, self.description(wrong.metadata(1024)))

    def test_google_legacy_collection_cannot_be_used_as_voyage(self):
        validate_collection(EmbeddingConfig("google", "gemini-embedding-001"), self.description(dimension=3072))
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_collection(EmbeddingConfig("voyage", "voyage-4", 1024), self.description(dimension=3072))
        with self.assertRaisesRegex(ValueError, "lacks embedding identity"):
            validate_collection(EmbeddingConfig("voyage", "voyage-4", 1024), self.description())


if __name__ == "__main__":
    unittest.main()
