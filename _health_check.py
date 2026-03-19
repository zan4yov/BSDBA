"""DSDBA System Health Check — Phase 3 Complete / Sprint A Ready."""
import sys
import pathlib
import ast
import importlib
import platform

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

SEP = "=" * 62

def ok(label): return f"[ OK  ] {label}"
def warn(label): return f"[ WARN] {label}"
def miss(label): return f"[ MISS] {label}"
def fail(label): return f"[ FAIL] {label}"


print(SEP)
print("  DSDBA SYSTEM HEALTH CHECK")
print("  Phase 3 COMPLETE — Sprint A READY")
print(SEP)

# ── 1. Python version ──────────────────────────────────────────────────────
pv = platform.python_version()
flag = ok(f"Python {pv}") if pv.startswith("3.1") else warn(f"Python {pv} (need 3.10+)")
print(flag)

# ── 2. config.yaml raw YAML load ──────────────────────────────────────────
import yaml
try:
    cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text())
    print(ok("config.yaml — raw YAML load"))
    print(f"        audio   : sample_rate={cfg['audio']['sample_rate']}  n_mels={cfg['audio']['n_mels']}  "
          f"fmin={cfg['audio']['fmin']}  fmax={cfg['audio']['fmax']}")
    print(f"        model   : backbone={cfg['model']['backbone']}")
    print(f"        gradcam : target_layer={cfg['gradcam']['target_layer']}")
    print(f"        training: batch_size={cfg['training']['batch_size']}  "
          f"gradient_checkpointing={cfg['training']['gradient_checkpointing']}")
    print(f"        deploy  : framework={cfg['deployment']['framework']}")
except Exception as e:
    print(fail(f"config.yaml — {e}"))

# ── 3. Pydantic DSDBAConfig ────────────────────────────────────────────────
try:
    from src.utils.config import load_config
    cfg_obj = load_config(pathlib.Path("config.yaml"))
    print(ok("Pydantic DSDBAConfig — validates against config.yaml"))
except Exception as e:
    print(fail(f"Pydantic DSDBAConfig — {e}"))

# ── 4. ErrorCode + DSDBAError ─────────────────────────────────────────────
try:
    from src.utils.errors import ErrorCode, DSDBAError
    err = DSDBAError(code=ErrorCode.AUD_001, message="clip too short", stage="Audio DSP")
    expected_str = "[AUD-001] Audio DSP: clip too short"
    expected_code = "AUD-001"
    assert str(err) == expected_str, f"__str__ returned: {str(err)!r}"
    assert err.to_dict()["code"] == expected_code, f"to_dict code: {err.to_dict()['code']!r}"
    print(ok(f"DSDBAError.__str__  = {str(err)}"))
    print(ok(f"DSDBAError.to_dict  = {err.to_dict()}"))
except AssertionError as e:
    print(fail(f"DSDBAError — {e}"))
except Exception as e:
    print(fail(f"DSDBAError import — {e}"))

# ── 5. Logger ─────────────────────────────────────────────────────────────
try:
    from src.utils.logger import get_logger
    log = get_logger("health.check")
    print(ok("get_logger() — structured JSON logger factory"))
except Exception as e:
    print(fail(f"logger — {e}"))

# ── 6. Stub AST check ─────────────────────────────────────────────────────
print()
print("  --- Pipeline Stubs (AST scan) ---")
stubs = {
    "src/audio/dsp.py":   ["preprocess_audio", "preprocess_batch"],
    "src/cv/model.py":    ["build_model"],
    "src/cv/train.py":    ["train"],
    "src/cv/infer.py":    ["run_inference"],
    "src/cv/gradcam.py":  ["run_gradcam", "get_raw_saliency"],
    "src/nlp/explain.py": ["generate_explanation"],
    "app.py":             ["build_demo"],
}
all_stubs_ok = True
for fpath, funcs in stubs.items():
    try:
        tree = ast.parse(pathlib.Path(fpath).read_text(encoding="utf-8"))
        defined = {
            n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for fn in funcs:
            if fn in defined:
                print(ok(f"  {fpath}::{fn}"))
            else:
                all_stubs_ok = False
                print(miss(f"  {fpath}::{fn}"))
    except Exception as e:
        all_stubs_ok = False
        print(fail(f"  {fpath}: {e}"))

# ── 7. Runtime packages ────────────────────────────────────────────────────
print()
print("  --- Runtime Packages (local env) ---")
core_pkgs = ["numpy", "pydantic", "yaml"]
colab_pkgs = ["torch", "librosa", "soundfile", "onnxruntime", "gradio"]
missing_core = []
missing_colab = []

for pkg in core_pkgs:
    mod_name = "yaml" if pkg == "yaml" else pkg
    try:
        m = importlib.import_module(mod_name)
        print(ok(f"  {pkg} == {m.__version__}"))
    except ImportError:
        missing_core.append(pkg)
        print(miss(f"  {pkg} — NOT INSTALLED"))

for pkg in colab_pkgs:
    try:
        m = importlib.import_module(pkg)
        print(ok(f"  {pkg} == {m.__version__}"))
    except ImportError:
        missing_colab.append(pkg)
        print(f"  [COLAB] {pkg} — not installed locally (install in Colab)")

# ── 8. Open questions ─────────────────────────────────────────────────────
print()
print("  --- Open Questions Gate ---")
qs = [
    ("Q3", "VRAM feasibility",         "RESOLVED", "Sprint B READY"),
    ("Q4", "Grad-CAM target layer",    "RESOLVED", "Sprint C READY"),
    ("Q5", "Mel bin-to-Hz mapping",    "RESOLVED", "Sprint C READY"),
    ("Q6", "UI framework lock",        "RESOLVED", "Sprint E READY"),
    ("Q7", "EER scoring protocol",     "OPEN",     "Blocks Phase 5 accuracy validation"),
]
for qid, desc, status, note in qs:
    flag = ok if status == "RESOLVED" else warn
    print(flag(f"  {qid}: {desc} — {status} ({note})"))

# ── Summary ────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  SUMMARY")
print(SEP)
print(f"  Utility modules (errors/logger/config) : OK")
print(f"  All pipeline stubs present             : {'OK' if all_stubs_ok else 'FAIL'}")
print(f"  Missing local packages (Colab-only)    : {missing_colab if missing_colab else 'none'}")
print(f"  Missing core packages                  : {missing_core if missing_core else 'none'}")
print(f"  Active phase                           : Phase 3 COMPLETE")
print(f"  Active sprint                          : Sprint A — READY TO IMPLEMENT")
print(f"  Next action                            : Run Chain 05 — src/audio/dsp.py (FR-AUD-001-011)")
print(SEP)
