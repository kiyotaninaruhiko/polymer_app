# 対応モデル一覧 📚

本アプリケーションで利用可能な全14種類のモデルを紹介します。

---

## 📊 クイックリファレンス

| カテゴリ | モデル数 | 特徴 |
|----------|:-------:|------|
| Numeric | 1 | 解釈性が高い数値記述子 |
| Fingerprint | 5 | 類似性検索に適したビットベクトル |
| Embedding | 8 | 深層学習ベースの分子表現 |

---

## 🎯 用途別おすすめモデル

| 用途 | おすすめモデル | 理由 |
|------|---------------|------|
| 物性予測（低分子） | RDKit 2D + Morgan FP | 解釈性が高く、実績豊富 |
| 物性予測（ポリマー） | Polymer FP + PolyNC | ポリマー構造に対応 |
| 類似性検索 | Morgan FP / AtomPair FP | 高速なビット演算 |
| 構造アラート | MACCS Keys | 既知の官能基を検出 |
| 転移学習 | MolCLR-GIN / ChemBERTa | 豊富な事前学習済みモデル |
| 3D構造考慮 | Uni-Mol | 立体構造を反映 |
| グラフ機械学習 | MolCLR-GIN/GCN | 事前学習済みGNN |

---

## 📈 Numeric Descriptors（数値記述子）

### RDKit 2D Descriptors
| 項目 | 内容 |
|------|------|
| Provider名 | `rdkit_2d` |
| 出力次元 | 16〜200 |
| 計算速度 | ◎ 最速 |
| 事前学習 | 不要 |
| ポリマー対応 | ❌ |

**説明**: 分子量、TPSA、LogP、水素結合ドナー/アクセプター数など、化学的に解釈可能な記述子を計算。QSAR/QSPR研究の基盤。

**パラメータ**:
- `descriptor_set`: `basic`（16種類）または `full`（200種類以上）

---

## 🔢 Fingerprints（フィンガープリント）

### Morgan Fingerprint (ECFP)
| 項目 | 内容 |
|------|------|
| Provider名 | `morgan_fp` |
| 出力次元 | 512〜4096 |
| 計算速度 | ◎ 高速 |
| 事前学習 | 不要 |
| ポリマー対応 | ❌ |

**説明**: Extended Connectivity Fingerprint (ECFP) とも呼ばれる。分子の局所構造をハッシュ化してビットベクトルに変換。

**パラメータ**:
- `radius`: 探索半径（デフォルト: 2）
- `n_bits`: ビット数（デフォルト: 2048）

---

### MACCS Keys
| 項目 | 内容 |
|------|------|
| Provider名 | `maccs_keys` |
| 出力次元 | 166 |
| 計算速度 | ◎ 最速 |
| 事前学習 | 不要 |
| ポリマー対応 | ❌ |

**説明**: MDL社が定義した166種類の構造キー。各ビットが特定の官能基や構造パターンの有無を表す。

---

### Atom Pair Fingerprint
| 項目 | 内容 |
|------|------|
| Provider名 | `atompair_fp` |
| 出力次元 | 512〜4096 |
| 計算速度 | ◎ 高速 |
| 事前学習 | 不要 |
| ポリマー対応 | ❌ |

**説明**: 分子内の原子ペアとその間のトポロジカル距離をエンコード。

---

### Topological Torsion Fingerprint
| 項目 | 内容 |
|------|------|
| Provider名 | `torsion_fp` |
| 出力次元 | 512〜4096 |
| 計算速度 | ◎ 高速 |
| 事前学習 | 不要 |
| ポリマー対応 | ❌ |

**説明**: 連続する4原子のパスをエンコード。回転可能結合周りの構造を捉える。

---

### Polymer Fingerprint (PFP)
| 項目 | 内容 |
|------|------|
| Provider名 | `polymer_fp` |
| 出力次元 | ~290 |
| 計算速度 | ◎ 高速 |
| 事前学習 | 不要 |
| ポリマー対応 | ✅ |

**説明**: ポリマー専用のフィンガープリント。モノマー記述子とMorgan FPを組み合わせ。

---

## 🤖 Embeddings（埋め込み）

### Transformer系

#### ChemBERTa-zinc
| 項目 | 内容 |
|------|------|
| Provider名 | `chemberta_zinc` |
| HuggingFace | `seyonec/ChemBERTa-zinc-base-v1` |
| 出力次元 | 768 |
| 計算速度 | △ 遅い |
| 事前学習 | ✅ 済 |
| ポリマー対応 | ✅ |

**説明**: ZINCデータセットで事前学習されたRoBERTaベースモデル。一般的な低分子向け。

---

#### ChemBERTa-pubchem
| 項目 | 内容 |
|------|------|
| Provider名 | `chemberta_pubchem` |
| HuggingFace | `seyonec/PubChem10M_SMILES_BPE_450k` |
| 出力次元 | 768 |
| 計算速度 | △ 遅い |
| 事前学習 | ✅ 済 |
| ポリマー対応 | ✅ |

**説明**: PubChem 10M分子で事前学習。より大規模なデータセットで学習。

---

#### MoLFormer
| 項目 | 内容 |
|------|------|
| Provider名 | `molformer` |
| HuggingFace | `ibm/MoLFormer-XL-both-10pct` |
| 出力次元 | 768 |
| 計算速度 | △ 遅い |
| 事前学習 | ✅ 済 |
| ポリマー対応 | ✅ |

**説明**: IBM開発の大規模分子Transformer。高精度な分子表現を生成。

---

#### PolyNC
| 項目 | 内容 |
|------|------|
| Provider名 | `polync` |
| HuggingFace | `hkqiu/PolyNC` |
| 出力次元 | 768 |
| 計算速度 | △ 遅い |
| 事前学習 | ✅ 済 |
| ポリマー対応 | ✅ |

**説明**: ポリマー専用のTransformerモデル。Natural Language + Chemical Language の統合学習。

> ⚠️ **注意**: Python 3.12+ と PyTorch 2.6+ が必要です。

---

### GNN系

#### GNN Embedding (GIN)
| 項目 | 内容 |
|------|------|
| Provider名 | `gnn_embed` |
| 出力次元 | 64〜512 |
| 計算速度 | ○ 中程度 |
| 事前学習 | ❌ ランダム初期化 |
| ポリマー対応 | ❌ |

**説明**: Graph Isomorphism Network。分子をグラフとして処理。ランダム初期化のため、事前学習済みモデル（MolCLR）の使用を推奨。

---

#### MolCLR-GIN
| 項目 | 内容 |
|------|------|
| Provider名 | `molclr_gin` |
| 出力次元 | 300 |
| 計算速度 | ○ 中程度 |
| 事前学習 | ✅ 済（対比学習） |
| ポリマー対応 | ❌ |

**説明**: Molecular Contrastive Learning of Representations。対比学習で事前学習されたGINモデル。高品質な分子埋め込みを生成。

**参考文献**: [MolCLR: Molecular Contrastive Learning of Representations via Graph Neural Networks](https://arxiv.org/abs/2102.10056)

---

#### MolCLR-GCN
| 項目 | 内容 |
|------|------|
| Provider名 | `molclr_gcn` |
| 出力次元 | 300 |
| 計算速度 | ○ 中程度 |
| 事前学習 | ✅ 済（対比学習） |
| ポリマー対応 | ❌ |

**説明**: MolCLRのGCN版。GINより軽量で高速。

---

### 3D構造系

#### Uni-Mol
| 項目 | 内容 |
|------|------|
| Provider名 | `unimol` |
| 出力次元 | 512 |
| 計算速度 | △ 遅い |
| 事前学習 | ✅ 済 |
| ポリマー対応 | ❌ |

**説明**: 3D構造を考慮した分子表現学習。立体構造が重要な予測タスクに有効。

---

## 📥 モデルのダウンロード

事前学習済みモデル（Transformer、MolCLR、Uni-Mol等）は初回使用時に自動ダウンロードされます。

キャッシュ場所:
- **Transformerモデル**: `~/.cache/huggingface/`
- **MolCLRモデル**: `~/.cache/polymer_app/molclr/`
