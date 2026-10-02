# KVKK and generative AI

!!! warning "Not legal advice"
    This page explains what wardcat does and does not change about sending
    text to a language model. It is not legal advice, and using wardcat does not
    on its own meet the requirements of KVKK (Law No. 6698) or any other regime. Decide with
    your data protection officer or counsel. [Türkçe sürüm](kvkk-generative-ai.tr.md).

## What the Authority says

**Cross-border transfers.** Article 9 of Law No. 6698 governs transfers of
personal data abroad. Law No. 7499 rewrote it in March 2024, in force from
1 June 2024. A transfer now rests on an adequacy decision, on appropriate
safeguards (such as standard contracts or binding corporate rules), or on
one of the incidental cases the article lists. The procedure is set by the
regulation published in the Official Gazette of 10 July 2024 (No. 32598).
KVKK's own [cross-border transfer page](https://www.kvkk.gov.tr/Icerik/2053/Yurtdisina-Aktarim)
has the current state, including any adequacy decisions.

**Generative AI.** KVKK published *Üretken Yapay Zekâ ve Kişisel Verilerin
Korunması Rehberi (15 Soruda)* on 24 November 2025
([page](https://www.kvkk.gov.tr/Icerik/8547/uretken-yapay-zeka-ve-kisisel-verilerin-korunmasi-rehberi-15-soruda)).
Two of its points bear on this page:

- **Question 10.** When a controller in Türkiye uses a generative AI system
  through a provider established abroad, and personal data goes abroad through
  it, the transfer has to comply with Article 9 and the 2024 regulation.
- **Anonymous data.** Data that is anonymous, or has been anonymised, is not
  personal data and falls outside the law. Whether data claimed to be
  anonymised really is has to be shown by technical methods and objective
  criteria. Until a data set is anonymised, it is still personal data.

The guide does not discuss pseudonymisation.

## What wardcat changes

- **The scan itself is not a transfer.** wardcat runs in your process. With
  the LLM layer pointed at a model on your own hardware, no text leaves your
  network to be scanned. A cloud PII service hosted outside Türkiye would make
  the scan itself a transfer.
- **Fewer identifiers in what you send.** Every value wardcat detects and
  replaces is absent from the prompt that goes to the provider.

## What it does not change

- **The rest of the text still goes abroad.** Anything wardcat does not
  detect, and anything under the `warn` action, reaches the provider as
  written. Detection is best effort: names, addresses and free-text details
  are missed at some rate. See [known limitations](limitations.md).
- **Tokens, hashes and surrogates are pseudonyms, not anonymisation.** The
  reversible actions exist so the values can come back. Whoever holds the
  salt or the token map can re-identify the text, and a deterministic hash of
  a low-entropy value (a phone number, a TC number) can be recovered by brute
  force. Treat a tokenised prompt as personal data unless you can show
  otherwise by the Authority's standard. wardcat does not establish that.
- **Context identifies.** "The CFO of a listed bank in Bursa, born in 1971"
  names a person with no name in it. Redacting identifiers does not remove
  that.
- **A realistic surrogate can be a real person's.** A surrogate TC number
  passes the checksum, and a surrogate name is an ordinary name. Either may
  belong to someone. See [security](security.md#surrogates).

## A starting configuration

```python
import os
from wardcat import Wardcat, Action

guard = (
    Wardcat(salt=os.environ["WARDCAT_SALT"])
    .with_preset("kvkk")        # a starting policy, not a compliance claim
    .with_ner(language="tr")
    .with_strict()              # a degraded scan raises instead of passing
)

result = guard.scan(text)
payload = result.reapply(Action.TOKENIZE)   # no `warn` value left in clear
answer = call_llm(payload.sanitized_text)
print(payload.restore(answer))
```

- Keep the salt, `token_map` and restored text on your side: out of logs,
  traces and the provider's reach. Log `result.redacted()`.
- Prefer `redact` for values you never need back.
- Review the [kvkk preset](presets.md) against your own data: what it leaves
  out is listed there.
