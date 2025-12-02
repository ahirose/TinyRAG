# TinyRAG - Simple RAG System

シンプルで軽量なRAG（Retrieval-Augmented Generation）システムの実装です。
A simple and lightweight RAG (Retrieval-Augmented Generation) system implementation.

## 🌟 特徴 / Features

- **シンプルなベクターDB**: Numpyベースの軽量なベクトルデータベース
- **簡単な統合**: 最小限のコードで始められます
- **柔軟なチャンキング**: オーバーラップするテキストチャンク機能
- **埋め込み生成**: Sentence Transformersによる高品質な埋め込み
- **コサイン類似度検索**: 効率的なセマンティック検索
- **OpenAI統合**: GPTモデルを使った回答生成

---

- **Simple Vector DB**: Lightweight numpy-based vector database
- **Easy Integration**: Get started with minimal code
- **Flexible Chunking**: Overlapping text chunks for better context
- **Embedding Generation**: High-quality embeddings via Sentence Transformers
- **Cosine Similarity Search**: Efficient semantic search
- **OpenAI Integration**: Answer generation with GPT models

## 📦 インストール / Installation

```bash
# リポジトリのクローン / Clone the repository
git clone <repository-url>
cd TinyRAG

# 依存関係のインストール / Install dependencies
pip install -r requirements.txt

# 環境変数の設定 / Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 🚀 クイックスタート / Quick Start

```python
from rag_system import SimpleRAG

# RAGシステムの初期化 / Initialize RAG system
rag = SimpleRAG(
    embedding_model="all-MiniLM-L6-v2",
    openai_model="gpt-3.5-turbo",
    chunk_size=500,
    chunk_overlap=50
)

# ドキュメントの追加 / Add documents
documents = [
    "Your first document text here...",
    "Your second document text here...",
]
rag.add_documents(documents)

# 検索 / Retrieve
results = rag.retrieve("Your question here", top_k=3)

# 回答生成 / Generate answer
answer = rag.generate_answer("Your question here", top_k=3)
print(answer['answer'])
```

## 📖 使用例 / Example

完全な使用例を実行:

```bash
python example.py
```

## 🏗️ アーキテクチャ / Architecture

```
┌─────────────────────────────────────────────────┐
│              SimpleRAG System                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. Document Loading                             │
│     └─> Text Chunking (with overlap)            │
│                                                  │
│  2. Embedding Generation                         │
│     └─> Sentence Transformers                   │
│                                                  │
│  3. Vector Storage                               │
│     └─> SimpleVectorDB (numpy)                  │
│                                                  │
│  4. Retrieval                                    │
│     └─> Cosine Similarity Search                │
│                                                  │
│  5. Answer Generation                            │
│     └─> OpenAI GPT Models                       │
│                                                  │
└─────────────────────────────────────────────────┘
```

## 🎯 実装されている機能 / Implemented Features

### 基本機能 / Core Features
- ✅ ドキュメントのチャンキング / Document chunking
- ✅ テキスト埋め込み生成 / Text embedding generation
- ✅ ベクター類似度検索 / Vector similarity search
- ✅ コンテキストベースの回答生成 / Context-based answer generation
- ✅ データベースの保存・読み込み / Database save/load
- ✅ メタデータサポート / Metadata support

### SimpleVectorDB 機能 / SimpleVectorDB Features
- ✅ ベクターの追加 / Add vectors
- ✅ コサイン類似度検索 / Cosine similarity search
- ✅ Top-k検索 / Top-k search
- ✅ データベースの永続化 / Database persistence
- ✅ クリア機能 / Clear functionality

## 🔮 提案する追加機能 / Proposed Additional Features

### 1. 高度な検索機能 / Advanced Search Features
- **ハイブリッド検索**: キーワード検索とセマンティック検索の組み合わせ
- **再ランキング**: クロスエンコーダーによる検索結果の再評価
- **フィルタリング**: メタデータベースのフィルタリング機能

### 2. ドキュメント処理 / Document Processing
- **マルチフォーマット対応**: PDF, Word, Markdown, HTMLのサポート
- **テーブル・画像抽出**: 構造化データの抽出機能
- **言語検出**: 多言語ドキュメントの自動検出

### 3. パフォーマンス最適化 / Performance Optimization
- **バッチ処理**: 大量ドキュメントの効率的な処理
- **キャッシング**: クエリ結果のキャッシュ機能
- **インデックス最適化**: FAISS等の高速ライブラリへの移行オプション

### 4. 品質向上機能 / Quality Improvement
- **回答評価**: 生成された回答の品質スコアリング
- **引用追跡**: 回答のソース追跡機能
- **信頼度スコア**: 検索結果の信頼度表示

### 5. ユーザーインターフェース / User Interface
- **Web UI**: Streamlit/Gradioベースのインターフェース
- **REST API**: FastAPIベースのAPIサーバー
- **CLI ツール**: コマンドラインインターフェース

### 6. 分析・モニタリング / Analytics & Monitoring
- **クエリログ**: 検索履歴の記録
- **使用統計**: システム使用状況の分析
- **パフォーマンスメトリクス**: 検索速度、精度の測定

### 7. セキュリティ / Security
- **アクセス制御**: ドキュメントレベルの権限管理
- **データ暗号化**: 保存データの暗号化
- **監査ログ**: アクセス履歴の記録

### 8. 高度なRAG技術 / Advanced RAG Techniques
- **Query Expansion**: クエリの自動拡張
- **Multi-hop Reasoning**: 複数ステップの推論
- **Self-RAG**: 自己評価による回答改善
- **Hypothetical Document Embeddings (HyDE)**: 仮想ドキュメントによる検索精度向上

## 📁 ファイル構成 / File Structure

```
TinyRAG/
├── README.md                 # このファイル / This file
├── requirements.txt          # 依存関係 / Dependencies
├── .env.example             # 環境変数テンプレート / Environment template
├── simple_vector_db.py      # ベクターDB実装 / Vector DB implementation
├── rag_system.py            # RAGシステム本体 / Main RAG system
└── example.py               # 使用例 / Usage example
```

## 🔧 設定 / Configuration

### RAGシステムパラメータ / RAG System Parameters

```python
SimpleRAG(
    embedding_model="all-MiniLM-L6-v2",  # 埋め込みモデル / Embedding model
    openai_model="gpt-3.5-turbo",        # OpenAIモデル / OpenAI model
    chunk_size=500,                       # チャンクサイズ / Chunk size
    chunk_overlap=50                      # チャンクオーバーラップ / Chunk overlap
)
```

### 利用可能な埋め込みモデル / Available Embedding Models

- `all-MiniLM-L6-v2` (推奨/Recommended): 高速で軽量 / Fast and lightweight
- `all-mpnet-base-v2`: より高精度 / Higher accuracy
- `paraphrase-multilingual-MiniLM-L12-v2`: 多言語対応 / Multilingual support

## 🤝 コントリビューション / Contributing

プルリクエストを歓迎します！
Pull requests are welcome!

## 📄 ライセンス / License

MIT License

## 🙏 謝辞 / Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) - 埋め込み生成 / Embedding generation
- [OpenAI](https://openai.com/) - 回答生成 / Answer generation
- [NumPy](https://numpy.org/) - ベクター演算 / Vector operations

## 📞 サポート / Support

問題が発生した場合は、GitHubのIssuesで報告してください。
For issues, please report on GitHub Issues.

---

**Note**: このシステムは教育・プロトタイピング目的で設計されています。本番環境での使用には、より堅牢なベクターデータベース（Pinecone, Weaviate, Qdrantなど）の使用を推奨します。

**Note**: This system is designed for educational and prototyping purposes. For production use, we recommend using more robust vector databases (Pinecone, Weaviate, Qdrant, etc.).
