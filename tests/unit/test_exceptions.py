"""Unit tests for veritascore.core.exceptions (Phase 0)."""

from __future__ import annotations

import pytest

from veritascore.core.exceptions import (
    ConfigurationError,
    DecompositionError,
    HardwareError,
    ModelLoadError,
    RetrievalError,
    VerificationError,
    VeritasCoreError,
)


EXCEPTION_CLASSES = [
    ModelLoadError,
    DecompositionError,
    VerificationError,
    RetrievalError,
    ConfigurationError,
    HardwareError,
]


class TestExceptionHierarchy:
    @pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
    def test_all_inherit_from_base(self, exc_class: type) -> None:
        assert issubclass(exc_class, VeritasCoreError)

    @pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
    def test_all_inherit_from_exception(self, exc_class: type) -> None:
        assert issubclass(exc_class, Exception)

    def test_base_is_exception(self) -> None:
        assert issubclass(VeritasCoreError, Exception)

    @pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
    def test_can_raise_and_catch_as_base(self, exc_class: type) -> None:
        with pytest.raises(VeritasCoreError):
            raise exc_class("test error")

    @pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
    def test_message_preserved(self, exc_class: type) -> None:
        msg = f"Error in {exc_class.__name__}"
        exc = exc_class(msg)
        assert str(exc) == msg

    def test_catch_specific_after_base(self) -> None:
        """Specific exceptions should be catchable independently."""
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("could not load model")

        # Should NOT be caught by a different subclass
        with pytest.raises(VeritasCoreError):
            try:
                raise ModelLoadError("model error")
            except DecompositionError:
                pass  # Should not reach here
