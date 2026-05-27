from __future__ import annotations

from pathlib import Path


# Patch scipy.sparse.eye_array (removed in scipy 1.15) for cvxpy compatibility
try:
    import scipy.sparse as _scipy_sparse

    if not hasattr(_scipy_sparse, "eye_array"):
        _scipy_sparse.eye_array = _scipy_sparse.eye  # type: ignore[attr-defined]
except Exception:
    pass


def _patch_rdagent_fin_factor() -> None:
    try:
        import os

        if os.environ.get("FORCE_LOCAL_STUB") == "1":
            import rdagent.oai.backend.base as _base

            def _fake_build_messages_and_create_chat_completion(self, *args, **kwargs):
                return '{"Momentum_10": {"description": "10-day momentum: (close - close.shift(10))/close.shift(10)", "formulation": "(close - close.shift(10))/close.shift(10)", "variables": {"close": "close"}, "hyperparameters": {"window": 10}}}'

            def _fake_try_create_chat_completion_or_embedding(self, *args, **kwargs):
                return '{"Momentum_10": {"description": "10-day momentum: (close - close.shift(10))/close.shift(10)", "formulation": "(close - close.shift(10))/close.shift(10)", "variables": {"close": "close"}, "hyperparameters": {"window": 10}}}'

            _base.APIBackend.build_messages_and_create_chat_completion = _fake_build_messages_and_create_chat_completion
            _base.APIBackend._try_create_chat_completion_or_embedding = _fake_try_create_chat_completion_or_embedding

        import rdagent.scenarios.qlib.experiment.utils as qlib_utils
        import rdagent.utils.env as rdagent_env
        from rdagent.oai.backend.base import LLM_SETTINGS
    except Exception:
        return

    try:
        retry_wait_seconds = int(os.environ.get("RDAGENT_RETRY_WAIT_SECONDS", "15"))
        if getattr(LLM_SETTINGS, "retry_wait_seconds", 0) < retry_wait_seconds:
            LLM_SETTINGS.retry_wait_seconds = retry_wait_seconds
    except Exception:
        pass

    # Reduce coding iterations per factor (10 → 3) — max_loop controls CoSTEER evolve
    try:
        qlib_utils.FACTOR_COSTEER_SETTINGS.max_loop = int(os.environ.get("RDAGENT_EVOLVING_N", "3"))
    except Exception:
        pass

    # Force docker env_type for backtest execution (conda not available in container)
    try:
        from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
        MODEL_COSTEER_SETTINGS.env_type = "docker"
    except Exception:
        pass

    # Patch QTDockerEnv.__init__ to fix extra_volumes for DIND on Mac.
    # Inside the outer container ~/.qlib = /root/.qlib, but the host Docker
    # daemon (used for DIND) needs the real Mac path (e.g. /Users/apple/.qlib).
    # Also adds the host project root so the cn_extra_data symlink resolves.
    #
    # Auto-detects HOST_HOME from the project root path so no env var is needed.
    try:
        _HOST_ROOT = str(Path(__file__).resolve().parent)
        # Derive host home from project root: /Users/apple/github/qlib-zh → /Users/apple
        _HOST_HOME = ""
        _root_parts = Path(_HOST_ROOT).parts
        if len(_root_parts) >= 3 and _root_parts[1] == "Users":
            _HOST_HOME = str(Path(*_root_parts[:3]))
        _HOST_HOME = os.environ.get("HOST_HOME", "") or _HOST_HOME

        _orig_qtdocker_init = rdagent_env.QTDockerEnv.__init__

        def _patched_qtdocker_init(self, conf=None, *args, **kwargs):
            if conf is None:
                conf = rdagent_env.QlibDockerConf()
            # Add host project root so cn_extra_data symlink resolves
            if _HOST_ROOT not in conf.extra_volumes:
                conf.extra_volumes[_HOST_ROOT] = {"bind": _HOST_ROOT, "mode": "ro"}
            # Replace /root/.qlib with the real host path for DIND
            # (only needed on Docker for Mac where /root/ is not shared)
            if _HOST_HOME:
                _host_qlib = os.path.join(_HOST_HOME, ".qlib")
                _old_key = str(Path("~/.qlib/").expanduser().resolve().absolute())  # "/root/.qlib"
                if _old_key in conf.extra_volumes and _old_key != _host_qlib:
                    conf.extra_volumes[_host_qlib] = conf.extra_volumes.pop(_old_key)
            _orig_qtdocker_init(self, conf=conf, *args, **kwargs)

        rdagent_env.QTDockerEnv.__init__ = _patched_qtdocker_init
    except Exception:
        pass

    # Inject optimized YAML templates into workspace at runtime
    # (host-side YAMLs, since the container has old defaults)
    try:
        import rdagent.scenarios.qlib.experiment.workspace as _rws
        _HOST_OPTIMIZED_YAML_DIR = Path(__file__).resolve().parent / "rdagent_workspace" / "factor_template_optimized"
        if _HOST_OPTIMIZED_YAML_DIR.exists():
            _orig_ws_init = _rws.QlibFBWorkspace.__init__

            def _patched_ws_init(self, template_folder_path, *args, **kwargs):
                _orig_ws_init(self, template_folder_path, *args, **kwargs)
                for yf in _HOST_OPTIMIZED_YAML_DIR.glob("*.yaml"):
                    target = self.workspace_path / yf.name
                    if target.exists():
                        target.write_text(yf.read_text())

            _rws.QlibFBWorkspace.__init__ = _patched_ws_init
    except Exception:
        pass

    original_prepare = rdagent_env.QTDockerEnv.prepare

    def _patched_prepare(self, *args, **kwargs):
        data_root = Path.home() / ".qlib" / "qlib_data" / "cn_data"
        if data_root.exists():
            try:
                from loguru import logger

                logger.info("Data already exists. Download skipped.")
            except Exception:
                pass
            return None
        return original_prepare(self, *args, **kwargs)

    def _patched_generate_data_folder_from_qlib():
        template_path = Path(__file__).resolve().parent / "rdagent_workspace" / "factor_data_template"

        # Skip Docker generation if HDF5 files already exist (pre-generated)
        if not (template_path / "daily_pv_all.h5").exists() or not (template_path / "daily_pv_debug.h5").exists():
            qtde = rdagent_env.QTDockerEnv()
            qtde.prepare()
            execute_log = qtde.check_output(local_path=str(template_path), entry="python generate.py")
            assert (template_path / "daily_pv_all.h5").exists(), (
                "daily_pv_all.h5 is not generated. Please check the log: \n" + execute_log
            )
            assert (template_path / "daily_pv_debug.h5").exists(), (
                "daily_pv_debug.h5 is not generated. Please check the log: \n" + execute_log
            )

        data_folder = Path(qlib_utils.FACTOR_COSTEER_SETTINGS.data_folder)
        data_folder.mkdir(parents=True, exist_ok=True)
        (data_folder / "daily_pv.h5").write_bytes((template_path / "daily_pv_all.h5").read_bytes())
        (data_folder / "README.md").write_text((template_path / "README.md").read_text())

        data_folder_debug = Path(qlib_utils.FACTOR_COSTEER_SETTINGS.data_folder_debug)
        data_folder_debug.mkdir(parents=True, exist_ok=True)
        (data_folder_debug / "daily_pv.h5").write_bytes((template_path / "daily_pv_debug.h5").read_bytes())
        (data_folder_debug / "README.md").write_text((template_path / "README.md").read_text())

    rdagent_env.QTDockerEnv.prepare = _patched_prepare
    qlib_utils.generate_data_folder_from_qlib = _patched_generate_data_folder_from_qlib

    # Stub embedding to avoid API calls (DeepSeek has no embedding endpoint)
    # Patch both base APIBackend and LiteLLMAPIBackend inner function
    try:
        import rdagent.oai.backend.base as _base_embed
        _DIM = 1536

        def _fake_create_embedding(self, input_content, *args, **kwargs):
            if isinstance(input_content, str):
                return [0.0] * _DIM
            return [[0.0] * _DIM for _ in input_content]

        _base_embed.APIBackend.create_embedding = _fake_create_embedding

        # Also stub LiteLLM backend's inner function which bypasses base class
        try:
            import rdagent.oai.backend.litellm as _litellm_mod
            _litellm_mod.LiteLLMAPIBackend._create_embedding_inner_function = lambda self, *a, **kw: [0.0] * _DIM
        except Exception:
            pass
    except Exception:
        pass


def _patch_factor_dedup() -> None:
    """Inject existing factors from new_factor.md + fail_new_factor.md into LLM prompts
    and hard-filter duplicates in convert_response, so rdagent doesn't waste tokens
    re-proposing factors that already exist or have already failed."""
    try:
        import json
        import os
        import re
    except Exception:
        return

    project_root = Path(__file__).resolve().parent
    existing_factor_names: set[str] = set()
    existing_lower: dict[str, str] = {}  # normalized lower -> original name

    def _normalize(name: str) -> str:
        return name.lower().replace("_", "").replace("-", "").replace(" ", "")

    # --- Gather existing factor names from new_factor.md ---
    nf_path = project_root / "tushare" / "new_factor.md"
    if nf_path.exists():
        for line in nf_path.read_text().split("\n"):
            s = line.strip()
            # Section header: ### 1. FactorName [optional tags]
            m = re.match(r"^###\s+\d+\.\s+(\S+)", s)
            if m:
                name = m.group(1)
                existing_factor_names.add(name)
                existing_lower[_normalize(name)] = name
            # Table row: | 1 | FactorName | ...
            if s.startswith("|") and s.endswith("|"):
                parts = [p.strip() for p in s.split("|")]
                if len(parts) >= 3 and parts[1].isdigit():
                    name = parts[2]
                    existing_factor_names.add(name)
                    existing_lower[_normalize(name)] = name

    # --- Gather from fail_new_factor.md ---
    ff_path = project_root / "tushare" / "fail_new_factor.md"
    if ff_path.exists():
        for line in ff_path.read_text().split("\n"):
            s = line.strip()
            # Section header: ## N. factor_name
            m = re.match(r"^##\s+\d+\.\s+(\S+)", s)
            if m:
                name = m.group(1)
                existing_factor_names.add(name)
                existing_lower[_normalize(name)] = name
            # Summary table row: | FactorName | ...
            if s.startswith("|") and not s.startswith("| 因子名称") and not s.startswith("|---"):
                parts = [p.strip() for p in s.split("|")]
                if len(parts) >= 2 and parts[1] and not parts[1].startswith("#") and not parts[1].startswith("因子"):
                    existing_factor_names.add(parts[1])
                    existing_lower[_normalize(parts[1])] = parts[1]

    if not existing_factor_names:
        return

    # Build the dedup instruction block
    factor_list = "\n".join(f"  - {name}" for name in sorted(existing_factor_names))
    dedup_rag = (
        "The following factors ALREADY EXIST in the factor library (either passed or failed) "
        "and must NOT be re-proposed:\n"
        f"{factor_list}\n\n"
        "CRITICAL: Do NOT propose any factor with the same name, formula, or concept as any of the above. "
        "Check your proposed factor names, formulations, and concepts carefully before outputting."
    )

    try:
        import rdagent.scenarios.qlib.proposal.factor_proposal as _fp_mod
    except Exception:
        return

    # ---- Layer 1a: Inject into QlibFactorHypothesisGen.prepare_context (hypothesis prompt) ----
    _orig_gen_prepare = _fp_mod.QlibFactorHypothesisGen.prepare_context

    def _patched_gen_prepare(self, trace):
        ctx, flag = _orig_gen_prepare(self, trace)
        existing_rag = ctx.get("RAG", "")
        ctx["RAG"] = (existing_rag + "\n\n" + dedup_rag) if existing_rag else dedup_rag
        return ctx, flag

    _fp_mod.QlibFactorHypothesisGen.prepare_context = _patched_gen_prepare

    # ---- Layer 1b: Inject into QlibFactorHypothesis2Experiment.prepare_context (experiment prompt) ----
    _orig_exp_prepare = _fp_mod.QlibFactorHypothesis2Experiment.prepare_context

    def _patched_exp_prepare(self, hypothesis, trace):
        ctx, flag = _orig_exp_prepare(self, hypothesis, trace)
        existing_rag = ctx.get("RAG")
        ctx["RAG"] = (existing_rag + "\n\n" + dedup_rag) if existing_rag else dedup_rag
        return ctx, flag

    _fp_mod.QlibFactorHypothesis2Experiment.prepare_context = _patched_exp_prepare

    # ---- Layer 2: Hard dedup in QlibFactorHypothesis2Experiment.convert_response ----
    _orig_convert = _fp_mod.QlibFactorHypothesis2Experiment.convert_response

    def _patched_convert(self, response, hypothesis, trace):
        exp = _orig_convert(self, response, hypothesis, trace)
        unique_tasks = []
        for task in exp.tasks:
            if _normalize(task.factor_name) not in existing_lower:
                unique_tasks.append(task)
        exp.tasks = unique_tasks
        return exp

    _fp_mod.QlibFactorHypothesis2Experiment.convert_response = _patched_convert


_patch_rdagent_fin_factor()
_patch_factor_dedup()
