from __future__ import annotations

import io
import os
import shlex
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_COMMAND_LOG = Path.cwd() / "command.log"


@dataclass(frozen=True)
class CommandLoggingConfig:
    enabled: bool
    log_file: Path


class _TeeStream(io.TextIOBase):
    def __init__(self, terminal: io.TextIOBase, log_handle: io.TextIOBase) -> None:
        self._terminal = terminal
        self._log_handle = log_handle

    @property
    def encoding(self) -> str | None:
        return getattr(self._terminal, "encoding", None)

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._terminal.write(text)
        self._log_handle.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._terminal, "isatty", lambda: False)())

    def writable(self) -> bool:
        return True


def parse_command_logging(argv: Sequence[str]) -> tuple[list[str], CommandLoggingConfig]:
    clean_args: list[str] = []
    enabled = True
    log_file = Path.cwd() / DEFAULT_COMMAND_LOG.name
    index = 0
    args = list(argv)
    while index < len(args):
        token = args[index]
        if token == "--no-log":
            enabled = False
            index += 1
            continue
        if token == "--log-file":
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise SystemExit("--log-file requires a path.")
            log_file = _resolve_path(args[index + 1])
            index += 2
            continue
        if token.startswith("--log-file="):
            value = token.split("=", 1)[1]
            if not value:
                raise SystemExit("--log-file requires a path.")
            log_file = _resolve_path(value)
            index += 1
            continue
        clean_args.append(token)
        index += 1
    return clean_args, CommandLoggingConfig(enabled=enabled, log_file=log_file)


@contextmanager
def command_log_context(
    argv: Sequence[str],
    *,
    config: CommandLoggingConfig,
    program_name: str = "ag-foundation",
) -> Iterator[Path | None]:
    if not config.enabled:
        yield None
        return

    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()
    command_text = shlex.join([program_name, *argv])

    with config.log_file.open("a", encoding="utf-8") as log_handle:
        prefix = "\n" if log_handle.tell() > 0 else ""
        log_handle.write(
            "\n".join(
                [
                    prefix + "=" * 80,
                    "Command Log",
                    "=" * 80,
                    f"Started   : {started_at.isoformat()}",
                    f"Command   : {command_text}",
                    f"CWD       : {Path.cwd()}",
                    f"PID       : {os.getpid()}",
                    f"Log file  : {config.log_file}",
                    "=" * 80,
                    "",
                ]
            )
        )
        log_handle.flush()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, log_handle)
        sys.stderr = _TeeStream(original_stderr, log_handle)
        exit_code = 0
        try:
            print(f"[logging] Appending command output to {config.log_file}")
            yield config.log_file
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1
            raise
        except BaseException as exc:
            exit_code = 1
            print("[logging] Command failed with an unhandled exception:", file=sys.stderr)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            raise
        finally:
            elapsed = time.perf_counter() - started_perf
            destination = sys.stderr if exit_code else sys.stdout
            print(
                (
                    "[logging] Finished "
                    f"(exit={exit_code}, finished={datetime.now().astimezone().isoformat()}, "
                    f"elapsed_seconds={elapsed:.2f})"
                ),
                file=destination,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()
