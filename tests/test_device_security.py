"""Prueba de seguridad de core/device.py — sandbox de rutas y bloqueo de secretos."""
import os
import sys
import shutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import core.device as device


def _ok(cond, label):
    print(("  OK  " if cond else "  XX  ") + label)
    if not cond:
        raise AssertionError(label)


def _first_existing_root() -> Path | None:
    """Primera ALLOWED_ROOT real donde podemos crear archivos de prueba."""
    for root in device.ALLOWED_ROOTS:
        try:
            if root.resolve().exists():
                return root.resolve()
        except OSError:
            continue
    return None


def main():
    print("\n[1] _resolve_safe rechaza rutas fuera del sandbox y traversal")
    # Una ruta claramente fuera de cualquier ALLOWED_ROOT (raíz del disco / sistema).
    outside = "C:\\Windows\\System32" if os.name == "nt" else "/etc"
    raised = False
    try:
        device._resolve_safe(outside)
    except PermissionError:
        raised = True
    _ok(raised, f"_resolve_safe lanza PermissionError fuera del sandbox ('{outside}')")

    # Traversal con '..' que escapa de una carpeta autorizada hacia un padre no autorizado.
    root = _first_existing_root()
    if root is not None:
        # Desde una raíz autorizada, subir lo bastante para salir de TODAS las raíces.
        traversal = str(root / ".." / ".." / ".." / ".." / "Windows")
        raised = False
        try:
            device._resolve_safe(traversal)
        except PermissionError:
            raised = True
        _ok(raised, "_resolve_safe lanza PermissionError ante traversal con '..'")
    else:
        print("  --  sin ALLOWED_ROOTS reales: omito la prueba de traversal con ruta real")

    print("\n[2] _is_secret distingue secretos de archivos normales (lógica pura)")
    _ok(device._is_secret(Path("/tmp/.env")), ".env se marca como secreto")
    _ok(device._is_secret(Path("/tmp/.env.production")), ".env.production se marca como secreto")
    _ok(device._is_secret(Path("/tmp/api.token")), "*.token se marca como secreto")
    _ok(device._is_secret(Path("/tmp/server.key")), "*.key se marca como secreto")
    _ok(device._is_secret(Path("/tmp/cert.pem")), "*.pem se marca como secreto")
    _ok(device._is_secret(Path("/tmp/my_secret_notes.txt")), "nombre con 'secret' se marca como secreto")
    _ok(device._is_secret(Path("/tmp/aws_credentials")), "nombre con 'credential' se marca como secreto")
    _ok(device._is_secret(Path("/tmp/PASSWORD.bak")), "'password' (mayúsculas) se marca como secreto")
    _ok(not device._is_secret(Path("/tmp/notas.txt")), "un .txt normal NO se marca como secreto")
    _ok(not device._is_secret(Path("/tmp/README.md")), "un .md normal NO se marca como secreto")

    print("\n[3] read_file bloquea secretos (PermissionError) y extensiones no permitidas (ValueError)")
    if root is None:
        print("  --  sin ALLOWED_ROOTS reales: omito con gracia las pruebas de read_file en disco")
        print("\nEXITO: DEVICE_SECURITY VALIDADO")
        return

    # Carpeta temporal DENTRO de una raíz autorizada (aislamiento + limpieza al final).
    tmp_dir = root / "_pentamodal_test_device_sec"
    created_dir = False
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        created_dir = True

        # (a) Secreto dentro de carpeta autorizada → PermissionError (no ValueError).
        env_file = tmp_dir / ".env"
        env_file.write_text("API_KEY=supersecreto\n", encoding="utf-8")
        raised_perm = False
        try:
            device.read_file(str(env_file))
        except PermissionError:
            raised_perm = True
        except ValueError:
            raised_perm = False
        _ok(raised_perm, "read_file lanza PermissionError sobre un .env en carpeta autorizada")

        # (b) Extensión no permitida (no secreta) → ValueError.
        bin_file = tmp_dir / "datos.bin"
        bin_file.write_bytes(b"\x00\x01\x02 binario no textual")
        raised_value = False
        try:
            device.read_file(str(bin_file))
        except ValueError:
            raised_value = True
        _ok(raised_value, "read_file lanza ValueError ante extensión no permitida ('.bin')")

        # (c) Control positivo: un .txt normal y autorizado SÍ se lee tal cual.
        txt_file = tmp_dir / "ok.txt"
        contenido = "contenido legible\n"
        txt_file.write_text(contenido, encoding="utf-8")
        _ok(device.read_file(str(txt_file)) == contenido,
            "read_file devuelve el contenido de un .txt autorizado")
    finally:
        if created_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nEXITO: DEVICE_SECURITY VALIDADO")


if __name__ == "__main__":
    main()
