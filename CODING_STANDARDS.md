# Generation AI Camp - コーディング規約

## 📋 概要

このプロジェクトは、PythonによるAI開発学習を目的としたワークスペースです。
統一されたコーディング規約により、読みやすく保守性の高いコードを目指します。

## 🛠️ 開発環境セットアップ

### 1. 必要な拡張機能のインストール

VSCodeで推奨拡張機能を自動インストールするか、以下を手動でインストール：

- **Python** (ms-python.python)
- **Flake8** (ms-python.flake8)
- **Black Formatter** (ms-python.black-formatter)
- **isort** (ms-python.isort)
- **MyPy Type Checker** (ms-python.mypy-type-checker)

### 2. Python環境のセットアップ

```bash
# 仮想環境の作成（プロジェクトごと）
python -m venv env

# 仮想環境の有効化
# Windows:
.\env\Scripts\Activate.ps1
# macOS/Linux:
source env/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# 開発用ツールのインストール
pip install flake8 black isort mypy pytest
```

## 📝 コーディング規約

### 基本ルール

- **最大行長**: 100文字
- **インデント**: スペース4文字
- **文字エンコード**: UTF-8
- **改行コード**: LF (Unix形式)

### コードフォーマット

#### 自動フォーマット
保存時に自動的に以下が実行されます：
- **Black**: コードフォーマット
- **isort**: import文の整理
- **Flake8**: 静的解析

#### 手動実行
```bash
# Black でフォーマット
black --line-length=100 .

# isort で import 整理
isort --profile=black .

# Flake8 でリント
flake8 .
```

### 命名規則

```python
# 変数・関数: snake_case
user_name = "example"
def get_user_info():
    pass

# クラス: PascalCase
class UserManager:
    pass

# 定数: UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3

# プライベート: 先頭に_
def _internal_method():
    pass
```

### ドキュメント

```python
def calculate_score(user_id: str, date: datetime) -> float:
    """ユーザーのスコアを計算する

    Args:
        user_id: ユーザーID
        date: 計算対象日時

    Returns:
        計算されたスコア値

    Raises:
        ValueError: 無効なユーザーIDの場合
    """
    pass
```

## 🧪 テストとチェック

### テスト実行
```bash
# 全テスト実行
pytest

# カバレッジ付き実行
pytest --cov=src --cov-report=html

# 特定ファイルのテスト
pytest tests/test_specific.py
```

### コード品質チェック
```bash
# Flake8でリント
flake8 src/

# MyPyで型チェック
mypy src/

# 全品質チェック
flake8 . && mypy src/ && pytest
```

## 📁 ディレクトリ構造

```
GenerationAiCamp/
├── .flake8                 # Flake8設定
├── pyproject.toml          # プロジェクト設定
├── .vscode/
│   ├── settings.json       # VSCode設定
│   └── extensions.json     # 推奨拡張機能
├── LessonXX/              # 各レッスン
│   ├── src/               # ソースコード
│   ├── tests/             # テストコード
│   ├── requirements.txt   # 依存関係
│   └── README.md          # レッスン説明
└── README.md              # このファイル
```

## 🚫 除外ファイル

以下は自動的にチェック対象から除外されます：
- `__pycache__/`
- `env/`, `venv/`
- `chroma_store/`
- `build/`, `dist/`
- `*.egg-info/`
- `.pytest_cache/`

## ⚡ トラブルシューティング

### よくある問題

1. **Flake8エラーが多すぎる場合**
   ```bash
   # 段階的修正
   flake8 --select=E9,F63,F7,F82 .
   ```

2. **Blackとの競合**
   ```bash
   # Black優先設定確認
   black --check --diff .
   ```

3. **import順序の問題**
   ```bash
   # isortで修正
   isort --check-only --diff .
   ```

## 📈 継続的改善

- 週次でコード品質レポートを確認
- 新しいルールの追加は全員で議論
- 設定ファイルの更新はプルリクエストで

---

**Happy Coding! 🐍✨**
