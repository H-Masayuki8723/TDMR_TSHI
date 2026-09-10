# Phase 2D: true ITI 0.35 の detector tap 長時間確認

## 目的と評価範囲

同じ送信データ・雑音を各 detector tap に与え、複数の独立したチャネルグリッドで比較する。
従来の 100-sector pilot と異なり、全 tap・全 SNR が揃った分だけを主集計に採用する。
pilot の CSV は混ぜない。候補の途中打ち切りも行わない。

対象は既存の K=15、LDPC n=1800/k=1730、総レート約 0.90104、4KB=32768情報bit/sector、
hard_codebook、BCJR equalizer 2反復、turbo 0 の構成。
この近似受信器と有限グリッド条件での最適候補を求める。物理的に普遍な最適tapや連続値の大域最小を保証しない。
内部符号、境界処理、チャネルモデル自体は変更しない。

## 実験計画

- true ITI: 0.35、両側のcross-track tapを固定。
- assumed ITI: 0.06、0.08〜0.16 (0.01刻み)、0.18、0.20、0.25、0.35。
- channel SNR: 18〜24 dB、1 dB刻み。
- 1チャンク100セクタ、100チャンク、各条件合計10,000セクタ。
- 14 tap × 7 SNR × 10,000 = 980,000 sector-detector evaluations。
- 同一チャンク/SNR内では全tapが同じデータと雑音を使用する。独立データは100チャンク×7 SNR分。
- 固定比較基準 tap=0.10。matched tap=0.35も同じ試行数で評価する。

過去の実測 36点/約6.86時間を単純比例すると全体は約1,867時間 (約78日)。
これは目安であり、tap/SNRによるLDPC反復、CPU、メモリ帯域、スレッド設定で大きく変わる。
最初の全条件500セクタは全体の5%、1,000セクタは10%。途中結果からも次の検討に進める。
長時間を許容しても、試行数だけで1e-8〜1e-10を実証できるわけではない。

## Windows Ubuntu / WSL 実行

リポジトリ更新後、Ubuntuのターミナルで実行する。新規依存ライブラリは不要。

```bash
cd ~/work/TDMR_TSHI
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python scripts/run_tap_search.py configs/phase2d_trueiti035_tap_search.yaml --prepare-only
tmux new -s tap035long
```

tmux内で長時間実行を開始する。

```bash
python -u scripts/run_tap_search.py configs/phase2d_trueiti035_tap_search.yaml
```

`Ctrl+B`、その後 `D` で切り離す。戻るときは `tmux attach -t tap035long`。
tmuxはWindowsのスリープ、シャットダウン、Windows Update再起動を防がない。
電源に接続し、解析中のスリープ設定と更新予定を確認する。

## メモリ・中断・再開

同時実行は1プロセス。1チャネルグリッド/SNRを処理したら子プロセスを終了し、メモリを解放する。
NumPy/BLASのスレッド数も1に固定する。10,000セクタ分の信号配列をまとめて確保しない。
ただし、100セクタのBCJR/LDPC配列自体は必要なので、任意の少容量PCでの動作保証ではない。

各tap完了直後にJSONを一時ファイルからatomic renameして保存する。
中断時に失うのは計算中のtapだけ。再開時のチャネル再生成にも時間はかかるが、完了tapの復号は繰り返さない。
通常終了、Ctrl+C、プロセス強制終了、再起動の後も、同じ実行コマンドで再開する。
同じキャンペーンの二重起動はOSロックで拒否する。古いロックファイルを手動削除しない。

チャンクごとのseedは(master_seed, chunk番号, SNR)から決定し、tapの実行順や中断に依存しない。
再開では未完了tapのみ計算するが、送信グリッドと雑音を同じseed・同じ100セクタで再生成する。
コード、依存ライブラリのバージョン、設定を変更した場合は再開を拒否する。
元に戻すか、新しい `--out-dir` を使い、異なる条件を混ぜない。
チェックポイントを削除するとその点は再計算されるため、解析中は編集・削除しない。

メモリ不足の場合は `sectors_per_chunk` を50にして `chunks` を200にする別キャンペーンを作る。
100と50では有限チャネルグリッド境界の割合も変わるため、既存100-sectorデータとは混ぜない。

## 出力と確認

既定の保存場所:

`outputs/tap_search/phase2d_trueiti035_tap_search_v1/`

`manifest.json` に設定・ソースハッシュ・環境を記録する。
`checkpoints/` は再開に必要な点ごとの元データ、`codebooks/` はキャンペーン専用のキャッシュ。
5チャンクごと (各条件500セクタ追加ごと) に `summaries/` を更新する。

- `raw_results.csv`: 保存済みの全点。途中チャンクも含む。
- `aggregate.csv`: 完了した共通チャンクだけのFSR、誤り数、試行数。単純なBER平均ではない。
- `observed_best_by_snr.csv`: 各SNRでの実測最小FSR。複数候補が同率なら全て残す。
- `paired_vs_reference.csv`: 同じチャンクのtap=0.10との差。負なら観測FSRが低い。
- `observed_target_brackets.csv`: 実測FSRが目標を跨ぐSNR区間。両側10失敗以上でのみlog線形補間。
- `extrapolation_sensitivity.csv`: 10失敗以上/点、3点以上でFSR上限0.1/0.3/0.5を比較する補助的外挿。
- `status.json`: 保存点数、全条件の完了チャンク数、各条件の均衡した試行数。

停止中に集計だけ更新するには同じコマンドへ `--summarize-only` を追加する。
走行中は更新済みCSVを参照する。1ジョブだけ追加して止めるには `--max-jobs 1` を指定できる。
独立した子プロセスから同じ保存先へ直接書き込まない。

## 統計上の読み方

sector間には同一チャネルグリッドやLDPCフレームによる依存があり得る。
主な差の不確かさは、独立チャンクをペアのまま再標本化するbootstrapで評価する。
最低5完了チャンクから出力するが、5だけで精密な推測ができるとは限らない。
これは点ごとの探索的95%区間で、多重比較を補正した最適性の証明や逐次停止用の信頼列ではない。
途中結果を見て採用する候補の最終検証は新しいseedで行う。

全チャンクで失敗ゼロ・差ゼロのbootstrap区間がゼロ幅でも、真のFSRや差が厳密にゼロとは言えない。
`degenerate_chunk_differences` を必ず確認する。
Wilson区間とゼロ失敗時の上限はsector独立仮定の参考値として列名に `iid` を付けている。
独立仮定で10,000セクタ0失敗でも、片側95%上限は約3e-4であり、1e-4以下の実証にはならない。

SNRごとに最良tapが異なる可能性がある。まず実測曲線と固定SNRでの差を確認し、
目標FSRでの観測区間、複数seedのばらつき、外挿上限を変えた感度の順で判断する。
探索端が最良、SNR範囲外でしか目標に到達しない、同率候補が多い場合は、
範囲拡張や新しいseedでの追加確認を行う。小数第2位の最適tap確定を自動宣言しない。
