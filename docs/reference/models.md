# Results & constants

## Results

::: wardcat.ScanResult

::: wardcat.Violation

::: wardcat.RedactedResult

::: wardcat.RestoredText

::: wardcat.Substitution

::: wardcat.UnrestoredValue

## Exceptions

Everything wardcat raises derives from `WardcatError`, so one `except` catches
the lot. `ConfigError` and `ModelDownloadError` also subclass the built-in they
replaced (`ValueError` / `RuntimeError`), so existing handlers keep working.

::: wardcat.WardcatError

::: wardcat.ContextMismatch

::: wardcat.ConfigError

::: wardcat.ModelDownloadError

::: wardcat.UnsupportedLanguageError

## Constants

::: wardcat.Entity

::: wardcat.Action

::: wardcat.Backend

::: wardcat.Language
