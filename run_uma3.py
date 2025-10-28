#!/usr/bin/env python3
"""
uma3.pyをルートディレクトリから実行するためのスクリプト

このスクリプトはGenerationAiCampのルートディレクトリから実行し、
Lesson25/uma3soft-app/src/uma3.pyを適切に起動します。
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """uma3.pyを実行するメイン関数"""

    # 現在のディレクトリを確認
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")

    # ルートディレクトリ（GenerationAiCamp）かどうか確認
    if not (current_dir / "GenerationAiCamp.code-workspace").exists():
        print("⚠️ このスクリプトはGenerationAiCampのルートディレクトリから実行してください")
        print(f"現在の場所: {current_dir}")
        print("正しい場所: C:\\work\\ws_python\\GenerationAiCamp")
        sys.exit(1)

    # uma3.pyのパスを設定
    uma3_path = current_dir / "Lesson25" / "uma3soft-app" / "src" / "uma3.py"

    # ファイルの存在確認
    if not uma3_path.exists():
        print(f"⚠️ uma3.pyが見つかりません: {uma3_path}")
        sys.exit(1)

    # 環境変数の設定（必要に応じて）
    env = os.environ.copy()
    env["PYTHONPATH"] = str(current_dir / "Lesson25" / "uma3soft-app" / "src")

    # uma3.pyを実行
    print(f"🚀 Starting uma3.py from root directory...")
    print(f"Target script: {uma3_path}")
    print(f"Working directory: {current_dir}")
    print("-" * 50)

    try:
        # uma3.pyを実行（現在のディレクトリはルートディレクトリのまま）
        result = subprocess.run(
            [sys.executable, str(uma3_path)],
            cwd=current_dir,  # ルートディレクトリで実行
            env=env,
            check=False
        )

        print(f"\n🏁 uma3.py finished with exit code: {result.returncode}")

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error running uma3.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()    main()
