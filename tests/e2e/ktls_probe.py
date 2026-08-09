"""Standalone kTLS diagnostic - not a test (no ``test_`` prefix, so pytest ignores it).

Reproduces what anycorn does for kTLS - a socket-backed ``wrap_socket`` server with
``SSL_OP_ENABLE_KTLS`` set - over a real loopback TCP connection, then reports whether the
kernel took over the TLS send path and which ``getsockopt`` probe reveals it. Run in CI to
tell, per interpreter, whether its OpenSSL actually offloads TLS to the kernel (and so
whether ``test_ktls_real`` can pass) rather than guessing. Exits 0 regardless.
"""

from __future__ import annotations

import socket
import ssl
import sys
import tempfile
import threading
from pathlib import Path

import trustme

# Kernel TLS ABI values not exported by CPython's socket module.
TCP_ULP = 31
SOL_TLS = 282
TLS_TX = 1


def _probe(sock: socket.socket, when: str) -> None:
    print(f"  [{when}]")
    try:
        ulp = sock.getsockopt(socket.IPPROTO_TCP, TCP_ULP, 16)
    except OSError as exc:
        print(f"    TCP_ULP getsockopt error: {exc!r}")
    else:
        print(f"    TCP_ULP = {ulp!r}")
    for size in (4, 40, 56, 128):
        try:
            value = sock.getsockopt(SOL_TLS, TLS_TX, size)
        except OSError as exc:
            print(f"    TLS_TX[{size}] error: {exc!r}")
        else:
            print(f"    TLS_TX[{size}] OK -> {len(value)} bytes")


def main() -> None:
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    print(f"OpenSSL: {ssl.OPENSSL_VERSION}")
    print(f"ssl.OP_ENABLE_KTLS present: {hasattr(ssl, 'OP_ENABLE_KTLS')}")
    try:
        print(
            f"tcp_available_ulp: {Path('/proc/sys/net/ipv4/tcp_available_ulp').read_text().strip()}"
        )
    except OSError as exc:
        print(f"tcp_available_ulp: unavailable ({exc!r})")

    if sys.platform != "linux":
        print("not linux; kTLS not applicable")
        return

    ca = trustme.CA()
    server_cert = ca.issue_cert("localhost", "127.0.0.1")
    tmp = Path(tempfile.mkdtemp())
    certfile, keyfile = tmp / "cert.pem", tmp / "key.pem"
    server_cert.cert_chain_pems[0].write_to_path(str(certfile))
    server_cert.private_key_pem.write_to_path(str(keyfile))

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(certfile), str(keyfile))
    if hasattr(ssl, "OP_ENABLE_KTLS"):
        server_ctx.options |= ssl.OP_ENABLE_KTLS
    client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ca.configure_trust(client_ctx)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve() -> None:
        conn, _ = listener.accept()
        with server_ctx.wrap_socket(conn, server_side=True) as tls:
            print(f"negotiated: {tls.version()} / {tls.cipher()}")
            _probe(tls, "after handshake, before write")
            tls.sendall(b"x" * 200_000)  # some builds enable kTLS TX lazily on first write
            _probe(tls, "after first write")

    thread = threading.Thread(target=serve)
    thread.start()
    with socket.create_connection(("127.0.0.1", port)) as raw:
        with client_ctx.wrap_socket(raw, server_hostname="localhost") as tls:
            received = 0
            while received < 200_000:
                chunk = tls.recv(65536)
                if not chunk:
                    break
                received += len(chunk)
    thread.join()
    print(f"client received {received} bytes")


if __name__ == "__main__":
    main()
