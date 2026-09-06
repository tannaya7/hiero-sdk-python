"""Test cases for the appendFile TCK handler and its parameter parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hiero_sdk_python.exceptions import PrecheckError, ReceiptStatusError
from hiero_sdk_python.file.file_id import FileId
from hiero_sdk_python.response_code import ResponseCode
from hiero_sdk_python.transaction.transaction import Transaction
from tck.handlers import registry
from tck.handlers.file import append_file
from tck.param.file import AppendFileParams
from tck.response.base import StatusOnlyResponse


pytestmark = pytest.mark.unit


def _mock_transaction():
    """A MagicMock standing in for FileAppendTransaction's fluent setter chain."""
    tx = MagicMock()
    tx.set_grpc_deadline.return_value = tx
    tx.set_file_id.return_value = tx
    tx.set_contents.return_value = tx
    tx.set_chunk_size.return_value = tx
    tx.set_max_chunks.return_value = tx

    receipt = MagicMock(status=ResponseCode.SUCCESS)
    tx.execute.return_value.get_receipt.return_value = receipt
    return tx


def test_append_file_is_registered():
    handler = registry.get_handler("appendFile")
    assert handler is not None and callable(handler)


def test_parse_json_params():
    """Only contents is required; maxChunks/chunkSize go through to_int."""
    params = AppendFileParams.parse_json_params(
        {
            "sessionId": "session-1",
            "fileId": "0.0.100",
            "contents": "hello world",
            "maxChunks": "5",
            "chunkSize": "1024",
        }
    )

    assert params.fileId == "0.0.100"
    assert params.contents == "hello world"
    assert params.maxChunks == 5
    assert params.chunkSize == 1024

    minimal = AppendFileParams.parse_json_params({"sessionId": "session-1", "contents": "hello"})
    assert minimal.fileId is None
    assert minimal.maxChunks is None
    assert minimal.chunkSize is None


def test_parse_json_params_requires_contents():
    """contents is spec-required; missing or non-string values must fail fast, not build an empty append."""
    with pytest.raises(ValueError):
        AppendFileParams.parse_json_params({"sessionId": "session-1"})

    with pytest.raises(ValueError):
        AppendFileParams.parse_json_params({"sessionId": "session-1", "contents": 123})


def test_append_file_wires_setters_and_returns_status():
    """contents/chunkSize must land before maxChunks, since chunk count derives from them."""
    params = AppendFileParams(
        sessionId="session-1", fileId="0.0.100", contents="hello world", maxChunks=5, chunkSize=1024
    )
    tx = _mock_transaction()

    with (
        patch("tck.handlers.file.get_client", return_value=MagicMock()),
        patch("tck.handlers.file.FileAppendTransaction", return_value=tx),
    ):
        result = append_file(params)

    assert isinstance(result, StatusOnlyResponse)
    assert result.status == "SUCCESS"

    tx.set_file_id.assert_called_once_with(FileId.from_string("0.0.100"))
    tx.set_contents.assert_called_once_with("hello world")
    tx.set_chunk_size.assert_called_once_with(1024)
    tx.set_max_chunks.assert_called_once_with(5)

    setter_order = [call[0] for call in tx.method_calls if call[0].startswith("set_")]
    assert setter_order.index("set_contents") < setter_order.index("set_max_chunks")
    assert setter_order.index("set_chunk_size") < setter_order.index("set_max_chunks")


def test_append_file_defaults_file_id_when_omitted():
    """An omitted fileId must default to FileId() so the network rejects it with INVALID_FILE_ID."""
    params = AppendFileParams(sessionId="session-1", contents="hello")
    tx = _mock_transaction()

    with (
        patch("tck.handlers.file.get_client", return_value=MagicMock()),
        patch("tck.handlers.file.FileAppendTransaction", return_value=tx),
    ):
        append_file(params)

    tx.set_file_id.assert_called_once_with(FileId())


def test_append_file_applies_common_transaction_params():
    """commonTransactionParams.signers has to reach freeze/sign via apply_common_params."""
    common_params = MagicMock()
    params = AppendFileParams(sessionId="session-1", contents="hello", commonTransactionParams=common_params)
    tx = _mock_transaction()
    client = MagicMock()

    with (
        patch("tck.handlers.file.get_client", return_value=client),
        patch("tck.handlers.file.FileAppendTransaction", return_value=tx),
    ):
        append_file(params)

    common_params.apply_common_params.assert_called_once_with(tx, client)


def test_append_file_propagates_receipt_failure():
    """A bad receipt status (e.g. non-existent file -> INVALID_FILE_ID) should raise, not be swallowed."""
    params = AppendFileParams(sessionId="session-1", fileId="0.0.999999", contents="hello")
    tx = _mock_transaction()
    tx.execute.return_value.get_receipt.side_effect = ReceiptStatusError(
        status=ResponseCode.INVALID_FILE_ID,
        transaction_id=None,
        transaction_receipt=MagicMock(),
        message="INVALID_FILE_ID",
    )

    with (
        patch("tck.handlers.file.get_client", return_value=MagicMock()),
        patch("tck.handlers.file.FileAppendTransaction", return_value=tx),
        pytest.raises(ReceiptStatusError),
    ):
        append_file(params)


def test_append_file_propagates_later_chunk_failure():
    """A later chunk failing at submission must surface, not get masked by an earlier chunk's SUCCESS.

    Uses a real FileAppendTransaction split into two 5-byte chunks so execute_all()
    actually submits twice; only Transaction.execute (the lower-level submission call)
    is mocked, first returning a successful chunk then raising on the second.
    """
    params = AppendFileParams(sessionId="session-1", fileId="0.0.100", contents="abcde12345", chunkSize=5)

    first_chunk_response = MagicMock()
    first_chunk_response.get_receipt.return_value = MagicMock(status=ResponseCode.SUCCESS)

    with (
        patch("tck.handlers.file.get_client", return_value=MagicMock()),
        patch.object(
            Transaction,
            "execute",
            side_effect=[
                first_chunk_response,
                PrecheckError(status=1, transaction_id="0.0.1@1.1", message="later chunk failed"),
            ],
        ),
        pytest.raises(PrecheckError),
    ):
        append_file(params)


def test_append_file_invalid_file_id_raises():
    """A malformed fileId should fail via FileId.from_string before hitting the network."""
    params = AppendFileParams(sessionId="session-1", fileId="not-a-file-id", contents="hello")

    with patch("tck.handlers.file.get_client", return_value=MagicMock()), pytest.raises(ValueError):
        append_file(params)
