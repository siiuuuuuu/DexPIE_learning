import pathlib
import sys


def load_r3m_model(modelid: str, pretrained: bool = True):
    try:
        from r3m import load_r3m
    except ModuleNotFoundError as exc:
        if exc.name != "r3m":
            raise

        repo_root = pathlib.Path(__file__).resolve().parents[3]
        vendored_r3m = repo_root / "third_party" / "r3m"
        if vendored_r3m.is_dir():
            sys.path.insert(0, str(vendored_r3m))

        try:
            from r3m import load_r3m
        except ModuleNotFoundError as retry_exc:
            if retry_exc.name == "r3m":
                raise ModuleNotFoundError(
                    "No module named 'r3m'. Install it with "
                    "`cd third_party/r3m && pip install -e .`, or keep the "
                    "vendored third_party/r3m directory at the repo root."
                ) from retry_exc
            raise

    return load_r3m(modelid, pretrained=pretrained)
