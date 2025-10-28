#!/usr/bin/env python3
"""
LINE Bot の設定確認とデバッグ用スクリプト

LINE Bot の設定状態を確認し、問題をデバッグするためのツールです。
"""

import os
import sys
from pathlib import Path


def main():
    """設定確認とデバッグのメイン処理"""

    print("=" * 60)
    print("LINE Bot 設定確認・デバッグツール")
    print("=" * 60)

    # 1. 環境変数の確認
    print("\n1. 環境変数の確認:")
    print("-" * 30)

    env_vars = [
        "LINE_ACCESS_TOKEN",
        "LINE_CHANNEL_SECRET",
        "OPENAI_API_KEY",
        "TO_USER_ID",
        "DEBUG_SKIP_SIGNATURE"
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            if var in ["LINE_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "OPENAI_API_KEY"]:
                print(f"✅ {var}: {'*' * 20}...{value[-4:]} (length: {len(value)})")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 未設定")

    # 2. .envファイルの確認
    print("\n2. .envファイルの確認:")
    print("-" * 30)

    env_file_path = Path("Lesson25/uma3soft-app/.env")
    if env_file_path.exists():
        print(f"✅ .envファイル存在: {env_file_path.absolute()}")

        try:
            with open(env_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.strip().split('\n')
            for line in lines:
                if '=' in line and not line.startswith('#'):
                    key = line.split('=')[0].strip()
                    if key in env_vars:
                        print(f"   📄 {key}: 設定済み")

        except Exception as e:
            print(f"❌ .envファイル読み込みエラー: {e}")
    else:
        print(f"❌ .envファイルが見つかりません: {env_file_path.absolute()}")

    # 3. ディレクトリ構造の確認
    print("\n3. ディレクトリ構造の確認:")
    print("-" * 30)

    important_paths = [
        "Lesson25/uma3soft-app/src/uma3.py",
        "Lesson25/uma3soft-app/src/reminder_schedule.py",
        "Lesson25/uma3soft-app/.env",
        "Lesson25/uma3soft-app/db/chroma_store"
    ]

    for path_str in important_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                print(f"✅ ファイル: {path}")
            else:
                print(f"✅ ディレクトリ: {path}")
        else:
            print(f"❌ 存在しません: {path}")

    # 4. 実行可能性のテスト
    print("\n4. 実行可能性のテスト:")
    print("-" * 30)

    # Python パッケージのインポートテスト
    packages_to_test = [
        "flask",
        "langchain_openai",
        "langchain_chroma",
        "linebot",
        "requests",
        "dotenv"
    ]

    for package in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {package}: インポート可能")
        except ImportError:
            print(f"❌ {package}: インポートエラー - pip install が必要")

    # 5. 推奨設定
    print("\n5. 推奨設定:")
    print("-" * 30)

    print("📋 推奨される設定手順:")
    print("   1. .envファイルを Lesson25/uma3soft-app/ に作成")
    print("   2. LINE_ACCESS_TOKEN と LINE_CHANNEL_SECRET を設定")
    print("   3. OPENAI_API_KEY を設定")
    print("   4. DEBUG_SKIP_SIGNATURE=true を設定（開発時のみ）")
    print()
    print("📋 実行コマンド例:")
    print("   python run_uma3.py")
    print("   または")
    print("   python Lesson25/uma3soft-app/src/uma3.py")
    print()
    print("📋 トラブルシューティング:")
    print("   - 400エラー: ターゲットID（グループID/ユーザーID）を確認")
    print("   - 401エラー: LINE_ACCESS_TOKEN を確認")
    print("   - 403エラー: Bot の権限を確認")
    print("   - 署名エラー: DEBUG_SKIP_SIGNATURE=true を設定（開発時）")

    print("\n" + "=" * 60)
    print("設定確認完了")
    print("=" * 60)

if __name__ == "__main__":
    main()    main()
