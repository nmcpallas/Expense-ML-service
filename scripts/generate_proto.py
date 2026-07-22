from app.proto_loader import ensure_proto_generated


if __name__ == "__main__":
    ensure_proto_generated()
    print("Generated Python gRPC files.")
