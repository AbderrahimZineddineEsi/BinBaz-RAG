# Saved runs

PDF exports of real searches from the app (print of the page, so what you see is what the app produced). Times are copied from the Pipeline tab of each export.

| File | Where it ran | Rewrite / judge | Query | Total time | Notes |
| --- | --- | --- | --- | --- | --- |
| [01-local-gtx1050-gemini-judge.pdf](runs/01-local-gtx1050-gemini-judge.pdf) | Local PC, GTX 1050 3 GB (embedder CPU, reranker GPU) | Gemini `gemini-3.5-flash-lite` rewrite and judge | `ما حكم صلاة الجماعة في المسجد؟` | not shown in the export | Results tab only. 2 × *Answers it*, 3 × *Related only*; the rewrite is `حكم صلاة الجماعة في المسجد`. |
| [02-local-gtx1050-jev-judge.pdf](runs/02-local-gtx1050-jev-judge.pdf) | Local PC, GTX 1050 3 GB | No rewrite; JEV `typesafe/jev-1.13` judge via OpenRouter | `كيف أكفر عن يمينٍ حلفتها ثم حنثت؟` | 6.72 s | Rerank 4.59 s, judge 1.15 s, cost $0.000261 for 5 judged fatwas. 3 × *Answers it*, 2 × *Related only*. |
| [03-colab-gpu-gemini-judge.pdf](runs/03-colab-gpu-gemini-judge.pdf) | Colab GPU | Gemini `gemini-3.5-flash-lite` rewrite and judge | `ما حكم صلاة الجماعة في المسجد؟` | 5.30 s | Rewrite 1.73 s, rerank 1.53 s, judge 1.95 s. 2 × *Answers it*, 3 × *Related only*. |
| [04-colab-gpu-jev-judge.pdf](runs/04-colab-gpu-jev-judge.pdf) | Colab GPU | Gemini `gemini-3.5-flash-lite` rewrite; JEV `typesafe/jev-1.13` judge | `ما حكم صلاة الجماعة في المسجد؟` | 2.94 s | Rewrite 0.98 s, rerank 1.61 s, JEV judge 0.34 s, cost $0.000190. |

## Local vs Colab

The Mizan rerank over 20 candidates took 4.59 s on the GTX 1050 and 1.53–1.61 s on the Colab GPU, about 3× faster. End to end, the two JEV runs took 6.72 s locally (no rewrite) and 2.94 s on Colab (with a rewrite step), so roughly half the time. These are single runs, and LLM latency varies a lot between calls, so treat the totals as indicative.
