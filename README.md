# scigym-rikyu-open-models

RIKYU の推論 API（OpenAI 互換）で提供される open-weight LLM を、SciGym（Duan et al. 2025）の公開コードの Controller と評価器をそのまま使って SciGym-small 137 件で評価した研究。論文の正本は main の `.research/latex/iclr2024/paper.pdf`、記録は `.research/record.json`（AIRAS の事前登録・検証ゲート付き）。

結果の要約: 完走した 5 モデル（deepseek-v4.1-flash、kimi-k2.6、kimi-k3、qwen3.6-35b、qwen3.8-27b）はすべて論文の GPT-4.1-mini（STE 0.6007）の水準に届き、4 モデルは論文 Table 1 の最良値（Gemini-2.5-Pro、STE 0.3212）も上回った。GLM 系 3 モデルは serving 速度の問題で打ち切り（記録の notes 参照）。

## 構成

| パス | 役割 |
|---|---|
| `config/config.yaml` | 共通設定。`max_iterations`（20）、`max_tokens`（32768）、`temperature`、`workers`（16）、`instance_timeout`（14400 秒）、`base_url`、`data_dir` |
| `config/run/<run_id>.yaml` | run ごとの設定。`model:` 必須。`workers:` と `instance_timeout:` は共通設定を上書きできる |
| `src/main.py` | 137 件を 1 件 1 プロセスで並列に走らせ、評価層の入力（`eval_inputs/scigym_small.json`）と `instances.tar.gz` を書く。3 周まで再試行、`resume/<run_id>/instances.tar.gz` があれば続きから |
| `src/train.py` | 公式 `Controller` を 1 件分動かす。OpenAI 互換 API の呼び出し、5xx の待ち直し、空応答（本文が `reasoning_content` に入る）の扱い、上限で切れた思考の呼び直し |
| `src/evaluate.py` | 評価層 `scigym_small` のレポートを `metrics.json` に写す |
| `Makefile` | `make run RUN_ID=<run_id> MODE=<sanity\|pilot\|full>`。実験 → airas-eval → metrics の順（AIRAS 管理、編集しない） |
| `Dockerfile`, `pyproject.toml`, `uv.lock` | aarch64（RIKYU）向けの固定環境。libroadrunner 2.7.0 / antimony 2.14.0 の差し替えと libcombine のスタブを含む |
| `.research/record.json` | 仮説・claim・run・結果・notes。追記のみ |
| `.research/results/<run_id>/` | 取り込んだ run の出力（5 ファイル）。`instances.tar.gz` に件ごとの会話・提出モデル・生の空応答が入る |

データは RIKYU 上の `/data1/rkp00041/scigym_improvement/data/scigym_sbml/small`（Hugging Face `h4duan/scigym-sbml` の small split と同一）。評価層が公式リリースの sha256 と照合する。

## 再利用の手引き

同じ実験を別のモデルや別の反復回数で行う agent 向け。AIRAS の記録は「run を投げる前に宣言したものだけが検証される」ので、順番が大事。

### 1. モデルを足す

1. `config/run/<run_id>.yaml` を作る。run_id は既存と重複しない名前（例 `glm-5.3-retry` ではなく、記録上は同じ run_id に結果を追記する方が筋が良い。既存 run_id を再実行する場合はこの手順の 2 を飛ばす）。

   ```yaml
   model: <API のモデル名>
   workers: 32           # 生成が遅いモデルだけ。API はバッチ処理で 1 要求の速度が落ちない
   instance_timeout: 43200  # 20 反復に 4 時間で足りないモデルだけ
   ```

2. 記録に claim と run を宣言する（`append_to_record`、`hypothesis_id: "h1"`）。雛形:

   ```json
   {"id": "c8",
    "statement": "<model> で SciGym-small 137 件を最大 20 反復で解かせたときの平均 STE は、論文の GPT-4.1-mini の 0.6007 を 0.05 より大きくは上回らない。",
    "rationale": "...", "verifier": {"kind": "seyval"},
    "criterion": {"metric": "trajectory_smape", "subject": "<run_id>", "reference": 0.6007, "op": "<=", "margin": 0.05, "reference_passage": "s1.p5"},
    "prediction": {"low": -0.3, "high": 0, "basis": "..."},
    "cites_passages": ["s1.p2", "s1.p3", "s1.p5", "s1.p10"],
    "designs": [{"id": "d8", "summary": "d1 と同じ設定でモデルを <model> に替える。",
                 "runs": [{"run_id": "<run_id>", "description": "...", "params": {"mode": "full"}, "cites_passages": ["s1.p3"]}],
                 "cites_passages": ["s1.p3", "s1.p4", "s2.p1"]}]}
   ```

   表に行を足すなら、同じ `key` の table を全行分で追記する（後の宣言が生きる）。

3. commit して `verify` に push（run はこの commit かその子孫で実行する）。
4. `dispatch_experiment(backend="seyval", branch_name="verify", run_id=<run_id>, run_stage="sanity"|"pilot"|"full", compute_id="byo:4f616e40-bb56-45aa-95b5-38fa4d2d0b8b", resource_count=1, time_limit="4-00:00:00")`。sanity は 1 件 2 反復、pilot は 10 件、full は 137 件。
5. 完了したら `import_run_outputs(run_stage="full")`（1 run ずつ）→ ローカルで pull → `update_record`（必ず最後）→ 論文に `\airasval{<run_id>.trajectory_smape}` などで数値を書く → `verify_paper_values(model=...)` で引用を判定 → もう一度 `update_record` → `verify` に push → CI が green なら main を fast-forward → CI が `paper.pdf` を main に載せる。

### 2. 反復回数を変える

`config/config.yaml` の `max_iterations` を変えると全 run の条件が変わるので、既存の run_id では宣言と矛盾する。別の run_id（例 `kimi-k3-iter40`）を新しい claim か design として宣言し、design の summary に反復回数を書く。`instance_timeout` は反復数に比例して延ばす（20 反復で 2〜4 時間）。`max_tokens` は 32768 のままでよい（上限で切れた思考は倍にして呼び直す）。

### 3. Seyval と RIKYU の設定

- Seyval に `register_repository` で登録し、`set_repository_env_var` で `RIKYU_API_KEY` を入れる（値は `.env` から。agent は `.env` を読めないことがあるので、利用者が `! grep '^RIKYU_API_KEY=' .env` で値を出す）。
- 計算資源は RIKYU の BYO（`byo:4f616e40-…`）、GPU は使わないが資源単位が GPU なので `resource_count=1`。時間上限は最大 `4-00:00:00`。
- 同時に走らせられる run は 5 本まで。
- Seyval は実行前に `.research/results` を空にする。失敗した run の件ごとの出力から続きを実行するときは、その run の `instances.tar.gz` を `resume/<run_id>/instances.tar.gz` に置いて `git add -f`（`.gitignore` の `*.gz` に当たる）で commit する。`inputs_from_runs` は completed の run しか指定できない。
- 進捗は RIKYU に ssh して `/data1/rkp00041/rku00122/<execution_id>/.research/results/<run_id>/instances/*/evaluation.json` を数えると分かる（Seyval のログには失敗した件しか出ない）。

### 4. 落とし穴

- **空応答**: thinking モデル（特に kimi-k2.6）は `finish_reason=stop` で `content` が空、回答が `reasoning_content` に入った応答を返す。`train.py` は最初の 1 回で reasoning の文を本文として渡す。生の応答は各件の `empty_responses.jsonl` に残る。
- **上限で切れた思考**: `finish_reason=length` なら `max_tokens` を倍（最大 131072）にして呼び直す。API は 131072 を受け付ける。
- **5xx**: RIKYU の gateway は 502 を断続的に返す。60 秒待って最大 30 回呼び直す。応答待ちは 1 時間。
- **ODE 内で止まる**: LLM のコードが硬い ODE を解いて C ライブラリ内で止まり、公式の 3 分制限（SIGALRM）が効かない。1 件 1 プロセスとプロセス単位の打ち切り（`instance_timeout`）が必須。3 回とも打ち切られた件は「提出なし」として不完全モデルで採点する（公式規約と同じ）。
- **取り込みの上限**: 取り込みは 1 ファイル 1 API 呼び出しで、800 ファイルを超えると GitHub の secondary rate limit を踏む。件ごとの出力は `instances.tar.gz` にまとめる（1 run 5 ファイル）。
- **airas 本体の更新**: CI は airas の main で values.tex を再生成する。CI が「values.tex differs」と言ったら、ローカルの airas を pull して `update_record` をやり直す。
- **PDF commit の後**: CI が `paper.pdf` を main に載せた後は、その main を第 1 親にして作業する。`verify` が分岐したら `git checkout -b x origin/main && git merge <verify の sha>` で merge する（record ゲートは paper.pdf を変える commit を、Publish Paper が build した commit の直後にしか認めない）。

### 5. 所要時間の目安（137 件、16 並列、20 反復）

| モデル | 生成速度（1 要求） | full の所要 |
|---|---|---|
| kimi-k3 | 約 50 tok/s | 8.6 時間 |
| kimi-k2.6 | 約 115 tok/s | 10.2 時間（空応答の呼び直しで入力 160M トークン） |
| qwen3.6-35b | — | 9.4 時間 |
| deepseek-v4.1-flash | — | 17.7 時間 |
| qwen3.8-27b | — | 20 時間 |
| glm-5.3-flash | 約 130 tok/s | 打ち切り（14 件 / 4 時間） |
| glm-5.2 | 約 55 tok/s | 打ち切り（13 件 / 4 時間） |
| glm-5.3 | 約 15 tok/s | 打ち切り（5 件 / 8 時間、48 並列でも） |

参考: 商用モデル（Vercel AI Gateway 経由、`auto-res2/scigym-table1-reproduction`）は GPT-4.1-mini 25 分、GPT-4.1 30 分、Gemini-2.5-Flash 50 分、Gemini-2.5-Pro 1.5 時間、4 run で約 200 USD。
