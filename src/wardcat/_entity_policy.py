"""Entity-policy concern of :class:`~wardcat.Wardcat`, factored out as a mixin.

This owns *what to detect and how*: the add/remove/change entity operations and
the read-side introspection, plus their helpers. It operates on the host's
``self._config`` dict and asks the host to rebuild via ``self._rebuild()`` — it
holds no detectors itself. Keeping it separate makes the entity rules a single,
cohesive unit and slims the ``Wardcat`` facade.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Self

from wardcat.core.models import KNOWN_ENTITY_TYPES, Action, Entity, warn_unknown_entity
from wardcat.core.registry import LAYER_ENTITIES, VALID_LAYERS
from wardcat.exceptions import ConfigError

logger = logging.getLogger("wardcat.guard")


class EntityPolicyMixin:
    """Entity configuration + introspection for ``Wardcat``.

    The composing class must provide ``_config`` (the config dict),
    ``_default_action_warned`` (a bool flag), and a ``_rebuild()`` method.
    """

    # Provided by the composing class (Wardcat) — declared for type checkers.
    _config: dict[str, Any]
    _default_action_warned: bool

    def _rebuild(self) -> None:  # pragma: no cover - overridden by Wardcat
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_type: str | Entity,
        action: str | Action | None = None,
        layers: list[str] | None = None,
    ) -> Self:
        """
        Enable a single entity type (or every type via :attr:`Entity.ALL`).

        Adding an entity always enables it — to turn one off, use
        :meth:`remove_entity`. Supports method chaining.

        :param entity_type: An :class:`~wardcat.Entity` constant (recommended,
                            e.g. ``Entity.EMAIL``) or its string form (``"EMAIL"``).
                            Pass :attr:`Entity.ALL` to enable **every** known
                            entity in one call, then prune with
                            :meth:`remove_entity`.
        :param action:      An :class:`~wardcat.Action` constant or its string
                            form. **When omitted, defaults to** ``"hash"`` and a
                            warning is logged (once per guard).
        :param layers:      Which detector layers should look for this entity —
                            any of ``"regex"``, ``"ner"``, ``"llm"``. ``None``
                            (default) uses every layer that supports the entity.
        :raises ConfigError: if ``entity_type`` is not a ``str``/``Entity``, the
                            ``action`` is invalid, or a ``layer`` is unknown.
        """
        action = self._action_or_default(action)
        if self._is_all(entity_type):
            for name in sorted(KNOWN_ENTITY_TYPES):
                self._set_entity(name, enabled=True, action=action, layers=layers)
        else:
            self._set_entity(entity_type, enabled=True, action=action, layers=layers)
        self._rebuild()
        return self

    def add_entities(
        self,
        entities: Iterable[str | Entity] | Mapping[str | Entity, Any],
        *,
        action: str | Action | None = None,
        layers: list[str] | None = None,
    ) -> Self:
        """
        Enable many entity types at once (single rebuild). Supports chaining.

        ``entities`` may be an iterable of names, a ``{name: action}`` mapping, or
        a ``{name: {"action": ..., "layers": ...}}`` mapping for per-entity
        control. :attr:`Entity.ALL` may appear as an entry. Any entity left
        without an action defaults to ``"hash"`` (warned once per guard).

        :raises ConfigError: if ``entities`` is not a mapping or iterable, or any
                            spec/action/entity is invalid.
        """
        if isinstance(entities, dict):
            specs = entities
        elif isinstance(entities, str | Entity):
            # A bare string/Entity is a common mistake — it would iterate chars.
            raise ConfigError(
                f"add_entities() expects a mapping or an iterable of entity types, "
                f"not a single {type(entities).__name__}. Use add_entity() for one entity."
            )
        else:
            try:
                specs = dict.fromkeys(entities, {})
            except TypeError as exc:
                raise ConfigError(
                    f"add_entities() expects a mapping or an iterable of entity types, "
                    f"got {type(entities).__name__}."
                ) from exc

        for name, spec in specs.items():
            if isinstance(spec, str):
                spec = {"action": spec}
            elif spec is None:
                spec = {}
            elif not isinstance(spec, dict):
                raise ConfigError(
                    f"Invalid spec for {name!r}: expected str, dict, or None, "
                    f"got {type(spec).__name__}."
                )
            entity_action = self._action_or_default(spec.get("action", action))
            entity_layers = spec.get("layers", layers)
            names = sorted(KNOWN_ENTITY_TYPES) if self._is_all(name) else [name]
            for n in names:
                self._set_entity(n, enabled=True, action=entity_action, layers=entity_layers)
        self._rebuild()
        return self

    def remove_entity(self, entity_type: str | Entity) -> Self:
        """
        Disable a single entity type (or every type via :attr:`Entity.ALL`).

        Removing an entity that was never enabled is a no-op. An unknown entity
        *name* (likely a typo) logs a warning, like :meth:`add_entity`. Chainable.

        :raises ConfigError: if ``entity_type`` is not a ``str`` or ``Entity``.
        """
        if self._is_all(entity_type):
            self._disable_all()
        else:
            name = self._normalize_entity(entity_type)
            self._warn_if_unknown(name)
            self._disable_entity(name)
        self._rebuild()
        return self

    def remove_entities(self, entities: Iterable[str | Entity]) -> Self:
        """
        Disable many entity types at once (single rebuild). Supports chaining.

        :raises ConfigError: if ``entities`` is not an iterable of entity types.
        """
        if isinstance(entities, str | Entity):
            raise ConfigError(
                f"remove_entities() expects an iterable of entity types, not a single "
                f"{type(entities).__name__}. Use remove_entity() for one entity."
            )
        try:
            items = list(entities)
        except TypeError as exc:
            raise ConfigError(
                f"remove_entities() expects an iterable of entity types, "
                f"got {type(entities).__name__}."
            ) from exc

        for entity_type in items:
            if self._is_all(entity_type):
                self._disable_all()
            else:
                name = self._normalize_entity(entity_type)
                self._warn_if_unknown(name)
                self._disable_entity(name)
        self._rebuild()
        return self

    def change_entity_action(self, entity_type: str | Entity, action: str | Action) -> Self:
        """
        Change the action of an entity that is **currently enabled**.

        Never enables a new entity: it only retargets the action of an active
        one. If the entity was removed or never added, it raises. The layers it
        runs on are unchanged. :attr:`Entity.ALL` changes every enabled entity.

        :raises ConfigError: if ``entity_type`` is not a ``str``/``Entity``, the
                            ``action`` is invalid, or the entity (or, for
                            ``Entity.ALL``, *any* entity) is not currently enabled.
        """
        action = self._normalize_action(action)

        if self._is_all(entity_type):
            active = self._active_entities()
            if not active:
                raise ConfigError(
                    "change_entity_action(Entity.ALL) failed: no entities are currently "
                    "enabled. Enable some first with add_entity()."
                )
            for name in sorted(active):
                self._apply_action(name, action)
        else:
            name = self._normalize_entity(entity_type)
            if not self._is_entity_active(name):
                raise ConfigError(
                    f"Cannot change action for {name!r}: it is not enabled "
                    "(it was removed or never added). Enable it first with add_entity()."
                )
            self._apply_action(name, action)
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    # Read API (introspection)
    # ------------------------------------------------------------------

    def enabled_entities(self) -> set[str]:
        """Return the set of entity types currently enabled on any detector layer."""
        return self._active_entities()

    def get_entity_action(self, entity_type: str | Entity) -> str | None:
        """Return the action of a currently-enabled entity, or ``None`` if not enabled.

        :raises ConfigError: if ``entity_type`` is not a ``str``/``Entity``, or is
                            :attr:`Entity.ALL` (use :meth:`entity_policy` instead).
        """
        if self._is_all(entity_type):
            raise ConfigError(
                "get_entity_action() does not accept Entity.ALL — use entity_policy()."
            )
        name = self._normalize_entity(entity_type)
        if not self._is_entity_active(name):
            return None
        return self._entity_action(name)

    def entity_policy(self) -> dict[str, str]:
        """Return a ``{entity_type: action}`` mapping of every enabled entity."""
        policy: dict[str, str] = {}
        for name in sorted(self._active_entities()):
            action = self._entity_action(name)
            if action is not None:
                policy[name] = action
        return policy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_all(entity_type: object) -> bool:
        """True if *entity_type* is the :attr:`Entity.ALL` sentinel (``Entity.All`` too)."""
        return entity_type is Entity.ALL or entity_type == Entity.ALL.value

    @staticmethod
    def _normalize_entity(entity_type: str | Entity) -> str:
        """Coerce an Entity/str to its canonical string value, validating the type."""
        if isinstance(entity_type, Entity):
            return entity_type.value
        if isinstance(entity_type, str):
            return entity_type
        raise ConfigError(f"entity_type must be a str or Entity, got {type(entity_type).__name__}.")

    def _action_or_default(self, action: str | Action | None) -> str | Action:
        """Return *action*, or the default ``hash`` (warned once) when unspecified."""
        if action is None:
            self._warn_default_action_once()
            return Action.HASH.value
        return action

    def _warn_default_action_once(self) -> None:
        """Warn at most once per guard that a missing action defaulted to ``hash``."""
        if self._default_action_warned:
            return
        self._default_action_warned = True
        logger.warning(
            "No action specified for one or more entities — defaulting to 'hash' (the safest "
            "action). Pass action=... (e.g. Action.WARN) to choose explicitly. This is logged "
            "once per guard."
        )

    @staticmethod
    def _normalize_action(action: str | Action) -> str:
        """Validate an Action/str against the action registry; return its name."""
        from wardcat.core.actions import registered_actions

        name = action.value if isinstance(action, Action) else action
        if not isinstance(name, str) or name not in registered_actions():
            raise ConfigError(
                f"Invalid action {action!r}. Registered actions: {sorted(registered_actions())}. "
                "Add one with wardcat.register_action(name, fn)."
            )
        return name

    def _warn_if_unknown(self, entity_type: str) -> None:
        """Log a warning if *entity_type* is neither a known nor a custom entity."""
        if entity_type not in self._config.get("custom_patterns", {}):
            warn_unknown_entity(entity_type)

    def _is_entity_active(self, entity_type: str) -> bool:
        """True if *entity_type* is currently enabled on any *active* detector layer."""
        ent = self._config.get("entities", {}).get(entity_type)
        if ent and ent.get("enabled"):
            return True
        # LLM entities only count when the LLM layer itself is enabled.
        llm_cfg = self._config.get("llm_detector", {})
        if llm_cfg.get("enabled"):
            llm = llm_cfg.get("entities", {}).get(entity_type)
            if llm and llm.get("enabled"):
                return True
        return False

    def _entity_action(self, entity_type: str) -> str | None:
        """Return the configured action for *entity_type*, or None if unset."""
        ent = self._config.get("entities", {}).get(entity_type)
        if ent and "action" in ent:
            return str(ent["action"])
        llm = self._config.get("llm_detector", {}).get("entities", {}).get(entity_type)
        if llm and "action" in llm:
            return str(llm["action"])
        return None

    def _active_entities(self) -> set[str]:
        """The set of entity types currently enabled on any layer."""
        names = set(self._config.get("entities", {})) | set(
            self._config.get("llm_detector", {}).get("entities", {})
        )
        return {name for name in names if self._is_entity_active(name)}

    def _apply_action(self, entity_type: str, action: str) -> None:
        """Update an entity's action on every layer where it exists (no rebuild)."""
        entities = self._config.get("entities", {})
        if entity_type in entities:
            entities[entity_type]["action"] = action
        llm_entities = self._config.get("llm_detector", {}).get("entities", {})
        if entity_type in llm_entities:
            llm_entities[entity_type]["action"] = action

    def _disable_entity(self, entity_type: str) -> None:
        """Set an entity's ``enabled`` flag to False on every layer (no rebuild)."""
        entities = self._config.get("entities", {})
        if entity_type in entities:
            entities[entity_type]["enabled"] = False
        llm_entities = self._config.get("llm_detector", {}).get("entities", {})
        if entity_type in llm_entities:
            llm_entities[entity_type]["enabled"] = False

    def _disable_all(self) -> None:
        """Disable every configured entity on every layer (no rebuild)."""
        for name in list(self._config.get("entities", {})):
            self._disable_entity(name)
        for name in list(self._config.get("llm_detector", {}).get("entities", {})):
            self._disable_entity(name)

    def _set_entity(
        self,
        entity_type: str | Entity,
        *,
        enabled: bool,
        action: str | Action,
        layers: list[str] | None,
    ) -> None:
        """Mutate config for one entity across the chosen layers (no rebuild)."""
        # (Entity.ALL must be expanded by the caller, never reach here.)
        if self._is_all(entity_type):
            raise ConfigError(
                "Entity.ALL cannot be set directly — it is expanded by add_entity()/"
                "add_entities() into the individual known entity types."
            )
        entity_type = self._normalize_entity(entity_type)
        action = self._normalize_action(action)
        self._warn_if_unknown(entity_type)

        if layers is None:
            target = [lyr for lyr, ents in LAYER_ENTITIES.items() if entity_type in ents]
            if not target:  # unknown / custom entity → default to regex
                target = ["regex"]
        else:
            invalid = set(layers) - VALID_LAYERS
            if invalid:
                raise ConfigError(
                    f"Invalid layer(s) {sorted(invalid)}. Valid: {sorted(VALID_LAYERS)}"
                )
            target = list(layers)

        # config["entities"] holds the action (always, so the engine can apply
        # it) and the shared enabled flag for the regex/NER layers (each
        # of which only fires when its own layer switch is on).
        uses_shared_map = ("regex" in target) or ("ner" in target)
        self._config.setdefault("entities", {})[entity_type] = {
            "enabled": enabled and uses_shared_map,
            "action": action,
        }
        # The LLM layer keeps its own enabled set.
        if "llm" in target:
            llm_entities = self._config.setdefault("llm_detector", {}).setdefault("entities", {})
            llm_entities[entity_type] = {"enabled": enabled, "action": action}
