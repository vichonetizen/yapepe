"""Prueba del token de auth local (core/local_auth.py): verificacion y entropia minima."""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.local_auth as la


def _ok(cond, label):
    print(("  OK  " if cond else "  XX  ") + label)
    if not cond:
        raise AssertionError(label)


def main():
    # ── Aislamiento: snapshot del estado global y del entorno ────────────────────
    # local_auth cachea el token en la global _token y puede leer/escribir
    # data/local.token. Reseteamos la cache y redirigimos _TOKEN_FILE a una ruta
    # temporal INEXISTENTE para no tocar el fichero real ni depender de el.
    saved_token = la._token
    saved_token_file = la._TOKEN_FILE
    saved_env = os.environ.get("LOCAL_TOKEN")

    tmp_token_file = la._TOKEN_FILE.parent / "_test_auth_token.nonexistent"
    try:
        if tmp_token_file.exists():
            tmp_token_file.unlink()
    except OSError:
        pass

    try:
        la._TOKEN_FILE = tmp_token_file

        print("\n[1] Generacion fuerte cuando no hay env ni fichero")
        # Sin LOCAL_TOKEN y con _TOKEN_FILE inexistente -> genera secrets.token_hex(32).
        os.environ.pop("LOCAL_TOKEN", None)
        la._token = None
        tok = la.get_token()
        _ok(isinstance(tok, str) and len(tok) == 64,
            f"get_token() genera un token de 64 hex (len={len(tok)})")
        _ok(all(c in "0123456789abcdef" for c in tok),
            "el token generado es hexadecimal en minusculas")
        _ok(la.get_token() == tok,
            "get_token() es estable: la cache devuelve el mismo token")

        print("\n[2] verify_token contra el token vigente")
        _ok(la.verify_token("") is False,
            "verify_token('') es False (candidato vacio)")
        _ok(la.verify_token("token-incorrecto") is False,
            "verify_token('token-incorrecto') es False")
        _ok(la.verify_token(la.get_token()) is True,
            "verify_token(get_token()) es True")

        print("\n[3] Un LOCAL_TOKEN corto (<32) NO se adopta")
        # Reseteamos la cache para forzar la re-evaluacion del entorno.
        short_env = "abc123"  # 6 chars, por debajo del minimo de 32
        os.environ["LOCAL_TOKEN"] = short_env
        la._token = None
        tok_short = la.get_token()
        _ok(tok_short != short_env,
            "el LOCAL_TOKEN debil se ignora (no se adopta tal cual)")
        _ok(len(tok_short) == 64,
            f"en su lugar se usa un token fuerte de 64 hex (len={len(tok_short)})")
        _ok(la.verify_token(short_env) is False,
            "verify_token(LOCAL_TOKEN corto) es False")

        print("\n[4] Un LOCAL_TOKEN fuerte (>=32) SI se adopta")
        # Caso de control que confirma el limite exacto: 32 chars se aceptan.
        strong_env = "x" * 32
        os.environ["LOCAL_TOKEN"] = strong_env
        la._token = None
        tok_strong = la.get_token()
        _ok(tok_strong == strong_env,
            "un LOCAL_TOKEN de 32 chars se adopta literalmente")
        _ok(la.verify_token(strong_env) is True,
            "verify_token(LOCAL_TOKEN fuerte) es True")

        print("\nEXITO: AUTH_TOKEN VALIDADO\n")
    finally:
        # ── Restaurar estado global y entorno ────────────────────────────────────
        la._token = saved_token
        la._TOKEN_FILE = saved_token_file
        if saved_env is None:
            os.environ.pop("LOCAL_TOKEN", None)
        else:
            os.environ["LOCAL_TOKEN"] = saved_env
        try:
            if tmp_token_file.exists():
                tmp_token_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
