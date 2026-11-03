"""Create and read a local encrypted OpenAI API config. Never prints the key."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / ".secrets" / "openai.enc.json"
AAD = b"promptcomp-openai-config-v1"


def _key(passphrase: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode())


def init_config(path: Path = DEFAULT_PATH) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if path.exists():
        raise FileExistsError(f"Config already exists: {path}")
    api_key = getpass.getpass("OpenAI API key (hidden): ").strip()
    if not api_key:
        raise ValueError("API key cannot be empty")
    passphrase = getpass.getpass("Config passphrase (hidden): ")
    if len(passphrase) < 12:
        raise ValueError("Use a passphrase of at least 12 characters")
    if passphrase != getpass.getpass("Repeat config passphrase: "):
        raise ValueError("Passphrases differ")
    salt, nonce = os.urandom(16), os.urandom(12)
    plaintext = json.dumps({"api_key": api_key}).encode()
    ciphertext = AESGCM(_key(passphrase, salt)).encrypt(nonce, plaintext, AAD)
    payload = {"format": "promptcomp-openai-aesgcm-v1", "salt": salt.hex(),
               "nonce": nonce.hex(), "ciphertext": ciphertext.hex()}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print(f"Encrypted config saved at {path}")


def load_config(path: Path = DEFAULT_PATH) -> dict[str, str]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    payload = json.loads(path.read_text())
    if payload.get("format") != "promptcomp-openai-aesgcm-v1":
        raise ValueError("Unsupported encrypted config format")
    passphrase = getpass.getpass("Config passphrase (hidden): ")
    plaintext = AESGCM(_key(passphrase, bytes.fromhex(payload["salt"]))).decrypt(
        bytes.fromhex(payload["nonce"]), bytes.fromhex(payload["ciphertext"]), AAD)
    config = json.loads(plaintext)
    if not isinstance(config.get("api_key"), str) or not config["api_key"]:
        raise ValueError("Encrypted config has no API key")
    return {"api_key": config["api_key"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init",))
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    init_config(args.path)
