from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from app.service import ExpenseClassifierService


def start_health_server(port: int, service: ExpenseClassifierService) -> ThreadingHTTPServer:
    """Expose Kubernetes-compatible health endpoints alongside the gRPC server."""

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {
                "/actuator/health",
                "/actuator/health/liveness",
                "/actuator/health/readiness",
            }:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            body = json.dumps(
                {
                    "status": "UP",
                    "components": {
                        "grpcService": {"status": "UP"},
                        "models": {
                            "status": "UP",
                            "trainedModels": service.model.trained_models_count,
                        },
                    },
                }
            ).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    Thread(target=server.serve_forever, daemon=True, name="health-server").start()
    return server
