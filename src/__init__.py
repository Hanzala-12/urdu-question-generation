"""Project package.

Make stdout/stderr UTF-8 so Urdu text prints on the Windows console
(cp1252 by default). No-op where the stream is already UTF-8 (Kaggle/Linux).
"""

import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
