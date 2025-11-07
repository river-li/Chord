# XTHP Attack Demo - Artifact Evaluation

## Prerequisites

```bash
uv sync
export OPENAI_API_KEY="your-api-key"
```

## How to Run Demos

**1. Targeted Semantic Hooking (`semantic_targeted_hooking.py`) - Section 4.B - Listing 2&3**

Demonstrates XTH (data harvesting) attack. CompanyToTicker malicious tool hooks YahooFinanceNews and exfiltrates company names. Shows baseline (LLM knows Tesla=TSLA) vs attack (malicious tool intercepts).

```bash
uv run demo/semantic_targeted_hooking.py
```

**2. Untargeted Semantic Hooking (`semantic_untargeted_hooking.py`) - Section 4.B - Listing 9**

Demonstrates XTH attack via error handling. ErrorExplainer hooks ANY tool that produces errors and exfiltrates system prompts, user prompts, and error messages.

```bash
uv run demo/semantic_untargeted_hooking.py
```

**3. Syntax Format Hooking (`syntax_format_hooking.py`) - Section 4.C - Listing 1**

Demonstrates XTP (data polluting) attack. YoutubeSearchPreprocessor hooks YouTubeSearch by claiming to format input. Malicious logic is server-side, selectively polluting search keywords (e.g., "Python programming" → "python security vulnerabilities").

```bash
uv run demo/syntax_format_hooking.py
```