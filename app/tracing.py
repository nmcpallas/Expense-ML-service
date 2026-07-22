from __future__ import annotations

import logging
import os
import re
import secrets
from contextvars import ContextVar
from typing import Callable

import grpc

TRACE_ID_HEADER = "x-trace-id"
_TRACE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(TraceIdFilter())
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s level=%(levelname)s%(trace_suffix)s %(name)s - %(message)s",
        handlers=[handler],
        force=True,
    )


def current_trace_id() -> str:
    return _trace_id.get()


def new_trace_id() -> str:
    return secrets.token_hex(16)


def trace_id_from_metadata(metadata: tuple[tuple[str, str], ...] | None) -> str:
    for key, value in metadata or ():
        if key.lower() == TRACE_ID_HEADER and _TRACE_ID_PATTERN.fullmatch(value):
            return value
    return new_trace_id()


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = current_trace_id()
        record.trace_suffix = "" if trace_id == "-" else f" traceId={trace_id}"
        return True


class TraceIdServerInterceptor(grpc.ServerInterceptor):
    """Restores the bot's trace ID for every unary gRPC request."""

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler

        trace_id = trace_id_from_metadata(handler_call_details.invocation_metadata)

        def traced_unary_unary(request: object, context: grpc.ServicerContext) -> object:
            token = _trace_id.set(trace_id)
            try:
                return handler.unary_unary(request, context)
            finally:
                _trace_id.reset(token)

        return grpc.unary_unary_rpc_method_handler(
            traced_unary_unary,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )
