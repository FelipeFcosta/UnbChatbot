"""Make `fine_tuning.inference` a proper Python package.

Importing `raft_inference` here ensures that Modal picks up the decorated
functions/classes whenever the package is imported (e.g. when using
`modal deploy -m fine_tuning.inference`).
"""

from . import raft_inference  # noqa: F401 