from __future__ import annotations

from pathlib import Path


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
        template_path = Path("/Users/apple/gitee/qlib/rdagent_workspace/factor_data_template")
        qtde = rdagent_env.QTDockerEnv()
        qtde.prepare()

        execute_log = qtde.check_output(local_path=str(template_path), entry="python generate.py")

        assert (template_path / "daily_pv_all.h5").exists(), (
            "daily_pv_all.h5 is not generated. It means rdagent_workspace/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
            + execute_log
        )
        assert (template_path / "daily_pv_debug.h5").exists(), (
            "daily_pv_debug.h5 is not generated. It means rdagent_workspace/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
            + execute_log
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


_patch_rdagent_fin_factor()
