# Known limitations

wardcat is a best-effort detector: false negatives and false positives are
expected, and using it does not by itself make a system compliant. The cases
below are known, and each has a mitigation. The [security policy](https://github.com/oguzhantopcu0/wardcat/blob/main/SECURITY.md)
lists the behaviours that are inherent to the design rather than bugs.

| Case | Description | Mitigation |
|---|---|---|
| Homoglyph domain | Confusable folding (`normalize_confusables`, on by default) catches Cyrillic/Greek lookalikes and fullwidth/Arabic digits (`ali@tеst.com`, `４111…`), but it is a curated skeleton, not the full Unicode table — an exotic lookalike may slip through | Add an allowlist/extra validation for high-stakes fields; the LLM layer can flag suspicious text |
| `US_ZIP_CODE` | A bare 5-digit ZIP is matched only with a `ZIP:` keyword; otherwise ZIP+4 (`12345-6789`) is required, to avoid false positives | Enable `POSTAL_CODE` (catches any bare 5-digit), or the LLM layer for contextual ZIPs |
| `EU_NATIONAL_ID` | Spanish DNI/NIE, French INSEE, Dutch BSN and Polish PESEL are matched by regex; German IDs are not | Enable the LLM layer — it catches German (and other) national IDs contextually |
| `PASSPORT` | Regex needs a passport keyword (`passport no:`, `pasaport`, `Reisepass`, `passeport`) — bare passport numbers are too generic to match safely | Enable the LLM layer — it catches unlabeled passport numbers |
| `FINANCIAL_AMOUNT` / `VAT_NUMBER` | `FINANCIAL_AMOUNT` is off by default; a bare Turkish Vergi No needs a keyword | Enable `FINANCIAL_AMOUNT` for confidential docs; use a `Vergi No`/`VKN` keyword or the LLM layer for VAT |
| Multilingual NER | One SpaCy model loads per language; wardcat bundles no language *detection* by design (keeps the core dependency-light) | Detect the language yourself (check `supported_languages()`) and pass several models via `language=[...]`, or use the language-agnostic LLM layer |
| European addresses | Regex needs a street-type keyword (Straße, Rue, Calle…); unnumbered informal addresses may be missed | Use the NER `ADDRESS` / LLM layer, or add a `custom_patterns` rule for your address format |
| Turkish NER (`tr_core_news_trf`) | The transformer model is incompatible with SpaCy 3.5+ | Use `tr_core_news_md` or `tr_core_news_lg` |
| Lower-cased text | Where the document carries no capitals, the "a name has a capital" rule is switched off, so a common-word sequence can surface as a `PERSON` | Preserve the original casing if you want the stricter rule, or drop the phrase with `add_allowlist([...])` / `with_llm(adjudicate=True)` |
| Weak checksums | The ABA routing, NHS and IMEI checks each let roughly one bare digit run in ten through, so an uncued match is scored `0.70` and the default floor leaves it alone | Write the number with its keyword (`IMEI: …`, `routing number …`), or call `with_min_confidence(0.6)` to act on uncued matches |
| Ethereum addresses | The `0x` + 40-hex form is checked on its shape; EIP-55 mixed-case checksumming needs keccak-256, which the standard library does not carry | Bitcoin addresses are fully checksum-verified; for Ethereum, pair the match with the LLM layer if a stricter check matters |
| NER types that moved | `GPE`/`LOC` used to be reported as `ADDRESS`, and `NORP` as `ORG`. A configuration written before that keeps working but stops covering them | Add `LOCATION` and/or `NRP` on the NER layer. The first scan of an affected guard says so once |
| Turkish NER quality | `tr_core_news_md/lg` are news-trained and may miss names in non-standard contexts | Combine with `.with_llm(...)` — the LLM catches names NER misses |
| Phone numbers outside TR/FR/DE | The built-in `PHONE` pattern deliberately refuses a bare `123 456 7890`; an unlabelled number in another national format falls through it | Name the regions you serve with [`with_phone_regions()`](configuration.md#phone-regions), or label the number (`Phone: …`) |
| Prompt injection | Scanned text is interpolated into the LLM prompt, so adversarial input can try to suppress the LLM layer's findings | Keep regex and NER in front of it; see [security](security.md#prompt-injection-llm-layer) |
