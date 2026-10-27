import os
import sqlite3
import tempfile
from pathlib import Path

import db


_TEMP_DATABASE_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="fintech-test-isolation-"
)
_TEMP_DATABASE_PATH = str(
    Path(_TEMP_DATABASE_DIRECTORY.name) / "default-test.db"
)
_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()
_ORIGINAL_CONNECT = sqlite3.connect
_SQLITE_CONNECTIONS = []

db.DB_PATH = _TEMP_DATABASE_PATH
for _function_name in (
    "get_db_connection",
    "init_db",
    "reset_db",
    "fetch_database_data",
):
    _function = getattr(db, _function_name)
    _defaults = list(_function.__defaults__ or ())
    if not _defaults:
        raise RuntimeError(
            f"Expected a database-path default for {_function_name}."
        )
    _defaults[0] = _TEMP_DATABASE_PATH
    _function.__defaults__ = tuple(_defaults)


def _resolve_sqlite_path(database):
    value = os.fsdecode(os.fspath(database))
    if value == ":memory:":
        return None
    if value.startswith("file:"):
        if ":memory:" in value or "mode=memory" in value:
            return None
        value = value[5:].split("?", maxsplit=1)[0]
    return Path(value).resolve()


def _guarded_connect(database, *args, **kwargs):
    resolved = _resolve_sqlite_path(database)
    if resolved is not None:
        try:
            common_path = Path(
                os.path.commonpath((str(_TEMP_ROOT), str(resolved)))
            )
        except ValueError as error:
            raise RuntimeError(
                f"SQLite database is outside temporary storage: {resolved}"
            ) from error
        if common_path != _TEMP_ROOT:
            raise RuntimeError(
                f"SQLite database is outside temporary storage: {resolved}"
            )
    _SQLITE_CONNECTIONS.append(str(database))
    return _ORIGINAL_CONNECT(database, *args, **kwargs)


sqlite3.connect = _guarded_connect


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep(
            "=",
            f"SQLite isolation: {_len_sqlite_connections()} guarded connections",
        )


def pytest_unconfigure(config):
    sqlite3.connect = _ORIGINAL_CONNECT
    _TEMP_DATABASE_DIRECTORY.cleanup()


def _len_sqlite_connections():
    return len(_SQLITE_CONNECTIONS)
