from __future__ import annotations

import sys
from pathlib import Path

from grpc_tools import protoc


ROOT_DIR = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT_DIR / "proto"
GENERATED_DIR = ROOT_DIR / "app" / "generated"
PROTO_FILE = PROTO_DIR / "expense_classifier.proto"


def ensure_proto_generated() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    pb2_file = GENERATED_DIR / "expense_classifier_pb2.py"
    pb2_grpc_file = GENERATED_DIR / "expense_classifier_pb2_grpc.py"

    if not pb2_file.exists() or not pb2_grpc_file.exists():
        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{PROTO_DIR}",
                f"--python_out={GENERATED_DIR}",
                f"--grpc_python_out={GENERATED_DIR}",
                str(PROTO_FILE),
            ]
        )
        if result != 0:
            raise RuntimeError(f"Failed to generate gRPC files from {PROTO_FILE}")

    generated_path = str(GENERATED_DIR)
    if generated_path not in sys.path:
        sys.path.insert(0, generated_path)
