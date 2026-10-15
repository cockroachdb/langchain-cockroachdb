"""Unit tests for vectorstore methods that don't require a database."""

import pytest

from langchain_cockroachdb.async_vectorstore import _check_batch_size


class TestBatchSizeValidation:
    """Test batch_size validation."""

    def test_max_batch_size_boundary(self) -> None:
        """Test that batch_size is capped at 16,383 (65,535 params / 4 params per row)."""
        # The limit is derived from CockroachDB's 65,535 parameter cap
        # Each inserted row uses 4 parameters
        max_allowed = 65535 // 4  # Should be 16,383

        # At the boundary should be OK
        _check_batch_size(max_allowed)

        # One over the boundary should fail
        with pytest.raises(
            ValueError,
            match=r"batch_size \d+ exceeds the maximum of 16383.*65,535 bound parameters",
        ):
            _check_batch_size(max_allowed + 1)

        # Way over should also fail
        with pytest.raises(
            ValueError,
            match=r"batch_size \d+ exceeds the maximum of 16383.*65,535 bound parameters",
        ):
            _check_batch_size(50000)


class TestIdNormalization:
    """Test id normalization in duplicate detection."""

    def test_duplicate_uuid_spellings_rejected(self) -> None:
        """Test that different spellings of the same UUID are rejected as duplicates."""
        import uuid

        from langchain_cockroachdb.async_vectorstore import AsyncCockroachDBVectorStore

        # Create a UUID and two spellings (lowercase and uppercase)
        u = str(uuid.uuid4())

        # Create a minimal instance with required methods
        class MinimalStore:
            batch_size = 100

            @staticmethod
            def _normalize_id(id_str):
                return AsyncCockroachDBVectorStore._normalize_id(id_str)

            def _validate_inputs(self, count, metadatas, ids):
                return AsyncCockroachDBVectorStore._validate_inputs(self, count, metadatas, ids)

        mock_store = MinimalStore()

        # This should raise ValueError before writing anything
        with pytest.raises(ValueError, match=r"[Dd]uplicate.*id"):
            AsyncCockroachDBVectorStore._prepare_rows(
                mock_store,
                texts=["text1", "text2"],
                embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                metadatas=[{}, {}],
                ids=[u.lower(), u.upper()],
                kwargs={},
            )
