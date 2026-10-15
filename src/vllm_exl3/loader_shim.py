"""Model-loader shim for standalone-saved NemotronH VL packs.

Some EXL3 packs store the text model under its standalone HF names
(``backbone.*``, ``lm_head.*`` at the checkpoint root) while vLLM's
NanoNemotronVL wrapper only accepts nested names
(``language_model.backbone.*``). The wrapper's load_weights silently drops
every tensor that does not start with ``language_model`` / ``mlp1`` /
``vision_model.radio_model.`` / ``sound``, so such packs load as an empty
shell and the first marker check fails.

This loader subclasses DefaultModelLoader and rewrites the weight-name
iterator before the model sees it:

  backbone.*            -> language_model.backbone.*
  lm_head.*             -> language_model.lm_head.*
  mtp.*                 -> dropped (upstream drops MTP weights anyway)
  vision_model.*        -> unchanged (accepted by the wrapper)
  mlp1.* / sound*        -> unchanged

No files are modified; the rewrite happens in the streaming iterator only.
"""

from __future__ import annotations

from collections.abc import Iterable

from vllm.model_executor.model_loader import (
    DefaultModelLoader,
    register_model_loader,
)


def _rewrite_standalone_nemotron_names(
    weights: Iterable[tuple[str, "object"]],
) -> Iterable[tuple[str, "object"]]:
    for name, tensor in weights:
        if name.startswith("backbone."):
            yield f"language_model.{name}", tensor
        elif name == "lm_head" or name.startswith("lm_head."):
            yield f"language_model.{name}", tensor
        elif name.startswith("mtp."):
            # Upstream's mapper drops MTP weights ("mtp": None); skip early so
            # the wrapper never sees them.
            continue
        else:
            yield name, tensor


@register_model_loader("exl3_standalone_nemotron")
class Exl3StandaloneNemotronLoader(DefaultModelLoader):
    """DefaultModelLoader with standalone NemotronH name rewriting."""

    def _prepare_weights(self, model_name_or_path, subfolder, revision, fall_back_to_pt, allow_patterns_overrides):
        # _prepare_weights switches on a fixed allowlist of load formats and
        # rejects anything unknown; this loader is a pure name rewrite over the
        # standard HF layout, so resolve the file set as "hf" and keep the
        # custom format only for the iterator rewrite below.
        saved = self.load_config.load_format
        try:
            self.load_config.load_format = "hf"
            return super()._prepare_weights(
                model_name_or_path, subfolder, revision, fall_back_to_pt, allow_patterns_overrides
            )
        finally:
            self.load_config.load_format = saved

    def _get_weights_iterator(self, source):
        it = super()._get_weights_iterator(source)
        return _rewrite_standalone_nemotron_names(it)
