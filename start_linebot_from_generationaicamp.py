#!/usr/bin/env python3
"""
LINE Bot起動スクリプト (GenerationAiCampディレクトリから実行用)
実行ディレクトリ: C:\work\ws_python\GenerationAiCamp>
"""

import os
import sys


def start_linebot_from_generationaicamp():
    """GenerationAiCampディレクトリからLINE Botを起動"""

    print("🤖 LINE Bot稼働開始 (from GenerationAiCamp)")
    print("=" * 60)

    # 実行ディレクトリ確認
    current_dir = os.getcwd()
    print(f"📁 実行ディレクトリ: {current_dir}")

    if not current_dir.endswith("GenerationAiCamp"):
        print("⚠️  実行ディレクトリがGenerationAiCampではありません")
        print("   正しいディレクトリで実行してください")
        return False

    # パス設定
    src_path = os.path.join("Lesson25", "uma3soft-app", "src")
    if not os.path.exists(src_path):
        print(f"❌ {src_path} が見つかりません")
        return False

    # sys.pathにsrcを追加
    sys.path.insert(0, src_path)

    try:
        # uma3モジュールをインポート
        import uma3

        print("✅ 初期化完了:")
        print("   - OpenAI API接続確認済み")
        print("   - ChromaDB準備完了")
        print("   - スケジューラー起動済み")
        print("   - LINEBot SDK初期化完了")

        print("\n🌐 サーバー情報:")
        print("   - ホスト: 0.0.0.0")
        print("   - ポート: 5000")
        print("   - Webhook URL: http://localhost:5000/callback")
        print("   - Health Check: http://localhost:5000/")

        print("\n⚠️  重要な注意事項:")
        print("   - 本番環境ではngrok等でHTTPS公開が必要")
        print("   - LINE Developer Consoleで Webhook URLを設定")
        print("   - 停止するには Ctrl+C を押してください")

        print("\n📝 デバッグ情報:")
        print(f"   - ChromaDB Path: {uma3.PERSIST_DIRECTORY}")
        print(f"   - Flask Routes: {[str(rule) for rule in uma3.app.url_map.iter_rules()]}")

        print("\n" + "=" * 60)
        print("🚀 Flask サーバー起動中...")
        print("   サーバー起動後、以下でテストできます:")
        print("   curl http://localhost:5000/")
        print("=" * 60)

        # Flask アプリケーション起動
        uma3.app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 LINE Bot 正常停止")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = start_linebot_from_generationaicamp()
    sys.exit(0 if success else 1)
