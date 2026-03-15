# Backchannel TTS Generator

バックチャネル用フレーズの音声を、**生成 → 確認 → 一括DL** の 3 ステップでデータセットとして取得できる Web アプリです。TTS は **MeloTTS を直接推論**（HTTP サーバー不要）し、全パラメータを画面から指定できます。

## 前提

- MeloTTS のインストール済み（[install.md](install.md) 参照）
- フレーズ定義用の JSON ファイル（後述）

## 起動方法

```bash
python -m backchannel_app.main --port 7860
```

または、インストール後に:

```bash
backchannel-app --port 7860
```

ブラウザで `http://localhost:7860` を開きます。**tts_server の起動は不要**です（アプリ内で melo.api.TTS を直接利用します）。

### オプション

| オプション | 説明 | デフォルト |
|-----------|------|------------|
| `--phrases` / `-p` | フレーズ JSON のパス | `config/backchannel_phrases.json` |
| `--port` / `-P` | 待ち受けポート | `7860` |
| `--host` / `-h` | バインド先ホスト | 未指定（Gradio のデフォルト） |
| `--share` / `-s` | 公開用リンクを発行 | オフ |

環境変数 `BACKCHANNEL_PHRASES_JSON` でフレーズ JSON パスを指定することもできます。

## フレーズ JSON の形式

JSON は **「フォルダ名（カテゴリ名_テキストのスネークケース）→ テキスト」** の辞書です。キーが ZIP 内の**フォルダ名**になり、その中に **生成時 ID**（`001.wav`, `002.wav`, …）で WAV が配置されます。

- キー: フォルダ名（ファイルシステムで有効な名前。スラッシュ不可）
- 値: 合成するテキスト

例:

```json
{
  "NLC_mm_hmm": "Mm-hmm.",
  "NLC_mhm": "Mhm.",
  "ACK_okay": "Okay.",
  "ACK_alright": "Alright."
}
```

ZIP をダウンロードすると、例えば次のような構成になります。

- `NLC_mm_hmm/001.wav`, `NLC_mm_hmm/002.wav`, …
- `NLC_mhm/001.wav`, …
- `ACK_okay/001.wav`, …

## UI の操作（3 ステップ）

1. **Step 1 — 生成**: 共通パラメータ（Language, Speaker, Speed, Sample rate, SDP ratio, Noise scale, Noise scale W）と「Patterns per phrase」を設定し、「Generate all phrases」を押す。JSON の全フレーズ × n パターンが生成され、一時ディレクトリに保存される。
2. **Step 2 — 確認**: フレーズをドロップダウンで切り替え、各フレーズの n パターンを試聴する。気に入らないスロットは「Regenerate 1」～「Regenerate n」で再生成でき、その場で上書きされる。
3. **Step 3 — 一括ダウンロード**: 「Download ZIP」を押すと、Step 1 で生成したデータセット（Step 2 で差し替えた分も含む）を 1 つの ZIP でダウンロードできる。

## 環境変数でのデフォルト値

起動前に次の環境変数を設定すると、UI の初期値として使われます。

| 環境変数 | 説明 | デフォルト |
|----------|------|------------|
| `MELOTTS_LANGUAGE` | 言語コード | `EN` |
| `MELOTTS_SPEAKER` | 話者名（言語ごとの `spk2id` で確認） | `EN-US` |
| `MELOTTS_SPEED` | 速度 | `1.0` |
| `MELOTTS_SAMPLE_RATE` | 出力 WAV のサンプルレート（Hz） | `44100` |
| `MELOTTS_SDP_RATIO` | SDP ratio | `0.2` |
| `MELOTTS_NOISE_SCALE` | Noise scale | `0.6` |
| `MELOTTS_NOISE_SCALE_W` | Noise scale W | `0.8` |
| `BACKCHANNEL_PHRASES_JSON` | フレーズ JSON のパス | `config/backchannel_phrases.json` |

## データセット取得の手順（3 ステップ）

1. 言語・話者・Speed・Sample rate・SDP/Noise を設定し、「Patterns per phrase」に 1 フレーズあたりのパターン数（例: 5）を入力。
2. **Step 1**: 「Generate all phrases」をクリック。全フレーズ × n パターンが生成される。
3. **Step 2**: フレーズを切り替えながら試聴し、気に入らないスロットは「Regenerate i」で再生成。
4. **Step 3**: 「Download ZIP」をクリックし、表示されたボタンから ZIP を保存。解凍すると、各フレーズのフォルダ名の下に `001.wav`, `002.wav`, … が並びます。
