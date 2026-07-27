from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time


PASSWORD_ITERATIONS = 310_000
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


@dataclass(frozen=True)
class AuthUser:
    id: int
    username: str
    role: str
    created_at: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at,
        }


class AuthStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_iterations INTEGER NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS sessions_expires_at_idx ON sessions(expires_at);
                """
            )

    def ensure_admin(self, username: str, password: str) -> AuthUser:
        existing = self.get_user(username)
        if existing:
            return existing
        return self.register(username, password, role="admin")

    def first_admin(self) -> AuthUser | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE role = 'admin' ORDER BY id ASC LIMIT 1"
            ).fetchone()
        return user_from_row(row) if row is not None else None

    def register(self, username: str, password: str, *, role: str = "member") -> AuthUser:
        clean_username = validate_username(username)
        validate_password(password)
        if role not in {"admin", "member"}:
            raise ValueError("Invalid account role")
        salt = secrets.token_bytes(16)
        password_hash = hash_password(password, salt, PASSWORD_ITERATIONS)
        created_at = int(time.time())
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (
                        username, password_hash, password_salt,
                        password_iterations, role, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_username,
                        password_hash.hex(),
                        salt.hex(),
                        PASSWORD_ITERATIONS,
                        role,
                        created_at,
                    ),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            raise ValueError("用户名已存在") from error
        return AuthUser(id=user_id, username=clean_username, role=role, created_at=created_at)

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        clean_username = str(username or "").strip()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                (clean_username,),
            ).fetchone()
        if row is None:
            # Keep missing-user timing close to a normal password check.
            hash_password(str(password or ""), b"\0" * 16, PASSWORD_ITERATIONS)
            return None
        expected = bytes.fromhex(str(row["password_hash"]))
        salt = bytes.fromhex(str(row["password_salt"]))
        actual = hash_password(str(password or ""), salt, int(row["password_iterations"]))
        if not hmac.compare_digest(actual, expected):
            return None
        return user_from_row(row)

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_digest(token), int(user_id), now, now + SESSION_TTL_SECONDS),
            )
        return token

    def user_for_session(self, token: str) -> AuthUser | None:
        if not token:
            return None
        now = int(time.time())
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT users.* FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ?
                """,
                (token_digest(token), now),
            ).fetchone()
        return user_from_row(row) if row is not None else None

    def revoke_session(self, token: str) -> None:
        if not token:
            return
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_digest(token),))

    def change_password(self, user_id: int, current_password: str, new_password: str) -> None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        if row is None:
            raise ValueError("账户不存在")
        user = self.authenticate(str(row["username"]), current_password)
        if user is None:
            raise ValueError("当前密码不正确")
        validate_password(new_password)
        salt = secrets.token_bytes(16)
        password_hash = hash_password(new_password, salt, PASSWORD_ITERATIONS)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, password_salt = ?, password_iterations = ?
                WHERE id = ?
                """,
                (password_hash.hex(), salt.hex(), PASSWORD_ITERATIONS, int(user_id)),
            )

    def get_user(self, username: str) -> AuthUser | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                (str(username or "").strip(),),
            ).fetchone()
        return user_from_row(row) if row is not None else None


def validate_username(username: str) -> str:
    clean = str(username or "").strip()
    if not USERNAME_PATTERN.fullmatch(clean):
        raise ValueError("用户名需为 3–32 位字母、数字、点、短横线或下划线")
    return clean


def validate_password(password: str) -> None:
    value = str(password or "")
    if len(value) < 8:
        raise ValueError("密码至少需要 8 位")
    if len(value) > 256:
        raise ValueError("密码不能超过 256 位")


def hash_password(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, int(iterations))


def token_digest(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def user_from_row(row: sqlite3.Row) -> AuthUser:
    return AuthUser(
        id=int(row["id"]),
        username=str(row["username"]),
        role=str(row["role"]),
        created_at=int(row["created_at"]),
    )
