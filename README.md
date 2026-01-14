# Polymer SMILES Descriptor Generator 🧬

ポリマーSMILESから記述子（descriptor / fingerprint / embedding）を生成するStreamlitアプリケーション。

## 機能

- **複数モデル対応**: RDKit 2D記述子、Morgan Fingerprint、Transformer Embedding、GNN等
- **ポリマーSMILES対応**: ワイルドカード（`*`）を含むポリマー表記に対応
- **共重合体入力**: モノマーSMILES＋モル組成比で入力可能
- **複数エクスポート形式**: CSV / Parquet / JSON
- **キャッシュ機能**: 同一設定での再実行を高速化

## クイックスタート

### 必要環境

- Python 3.11+ （PolyNCを使用する場合は Python 3.12+）
- pip

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/kiyotaninaruhiko/polymer_app.git
cd polymer_app

# 仮想環境を作成
python -m venv venv

# 仮想環境を有効化
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt

# アプリを起動
streamlit run app.py
```

ブラウザで http://localhost:8501 にアクセスしてください。

### Docker を使用する場合（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/kiyotaninaruhiko/polymer_app.git
cd polymer_app

# Docker Composeで起動
docker compose up -d

# ログを確認
docker compose logs -f
```

ブラウザで http://localhost:8501 にアクセスしてください。

```bash
# 停止
docker compose down
```

## 使い方

### 1. Input SMILES
- テキストエリアにSMILESを入力（1行1SMILES）
- CSVフォーマット（`id,smiles`）も対応
- ファイルアップロードも可能

### 2. Select Models
- カテゴリタブ（Numeric / Fingerprint / Embedding）からモデルを選択
- 複数モデルを同時選択可能

### 3. View Results
- 結果をテーブルで確認
- CSV/Parquet/JSONでダウンロード

## 対応モデル一覧（全14種類）

> 📖 各モデルの詳細は [MODELS.md](MODELS.md) を参照してください。

| カテゴリ | モデル | Provider名 |
|----------|--------|-----------|
| **Numeric** | RDKit 2D | `rdkit_2d` |
| **Fingerprint** | Morgan FP | `morgan_fp` |
| | MACCS Keys | `maccs_keys` |
| | AtomPair FP | `atompair_fp` |
| | Torsion FP | `torsion_fp` |
| | Polymer FP | `polymer_fp` 🔗 |
| **Embedding** | ChemBERTa-zinc | `chemberta_zinc` 🔗 |
| | ChemBERTa-pubchem | `chemberta_pubchem` 🔗 |
| | MoLFormer | `molformer` 🔗 |
| | PolyNC | `polync` 🔗 |
| | GNN (GIN) | `gnn_embed` |
| | MolCLR-GIN | `molclr_gin` |
| | MolCLR-GCN | `molclr_gcn` |
| | Uni-Mol | `unimol` |

🔗 = ポリマーSMILES対応

## プロジェクト構成

```
polymer_app/
├── app.py              # Streamlit UI
├── config.py           # 設定・モデルプリセット
├── requirements.txt    # 依存パッケージ
├── core/
│   ├── parsing.py      # SMILES解析・検証
│   └── cache.py        # キャッシュ機能
├── providers/
│   ├── base.py         # Provider抽象クラス
│   ├── registry.py     # プロバイダー登録
│   ├── rdkit2d.py      # RDKit 2D記述子
│   ├── morgan.py       # Morgan Fingerprint
│   ├── maccs.py        # MACCS Keys
│   ├── atompair.py     # AtomPair/TopologicalTorsion FP
│   ├── polymer_fp.py   # Polymer Fingerprint
│   ├── transformer_embed.py  # Transformer埋め込み
│   ├── gnn_embed.py    # GNN埋め込み
│   ├── molclr.py       # MolCLR事前学習済みGNN
│   └── unimol.py       # Uni-Mol 3D埋め込み
├── export_io/
│   └── export.py       # CSV/Parquet/JSON出力
└── tests/              # ユニットテスト
```

## テスト実行

```bash
python -m pytest tests/ -v
```

## ライセンス

MIT

## トラブルシューティング

### RDKitがインストールできない
```bash
pip install rdkit
```
それでもダメな場合はcondaを使用:
```bash
conda install -c conda-forge rdkit
```

### PolyNCでエラーが出る
Python 3.12とPyTorch 2.6+が必要です:
```bash
# Python 3.12環境を作成
python3.12 -m venv venv312
source venv312/bin/activate  # or .\venv312\Scripts\activate on Windows
pip install -r requirements.txt
pip install 'torch>=2.6.0'
```

### ポート8501が使用中
```bash
streamlit run app.py --server.port 8502
```
