#!/usr/bin/env python3
"""
今週の予定検索機能のテストスクリプト

ChromaDBの今週の予定検索機能をテストします。
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# パスの設定
sys.path.append(str(Path(__file__).parent / "Lesson25" / "uma3soft-app" / "src"))


def test_weekly_schedule():
    """今週の予定検索機能をテスト"""

    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from uma3_chroma_improver import Uma3ChromaDBImprover

        print("=" * 60)
        print("今週の予定検索機能テスト")
        print("=" * 60)

        # ChromaDBの初期化
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        persist_directory = "Lesson25/uma3soft-app/db/chroma_store"
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

        # Uma3ChromaDBImproverの初期化
        improver = Uma3ChromaDBImprover(vector_db)

        print(f"📂 ChromaDB path: {persist_directory}")
        print(f"📅 テスト日時: {datetime.now()}")

        # 今週の範囲を表示
        current_date = datetime.now()
        days_since_monday = current_date.weekday()
        monday = current_date.date() - timedelta(days=days_since_monday)
        sunday = monday + timedelta(days=6)

        print(f"📅 現在日時: {current_date}")
        print(f"📅 今週の範囲: {monday} ～ {sunday}")
        print()

        # 基本テスト：全期間の今週の予定
        print("🔍 基本テスト: 全期間の今週の予定")
        print("-" * 40)
        all_weekly_events = improver.get_weekly_schedule(current_date, future_only=False)
        print(f"📊 全期間の今週の予定: {len(all_weekly_events)}件")

        # 未来のみテスト：現在日時以降の今週の予定
        print("\n🔍 未来のみテスト: 現在日時以降の今週の予定")
        print("-" * 40)
        future_weekly_events = improver.get_weekly_schedule(current_date, future_only=True)
        print(f"📊 現在日時以降の今週の予定: {len(future_weekly_events)}件")

        if future_weekly_events:
            for j, result in enumerate(future_weekly_events[:3], 1):
                event_date = result.metadata.get("event_date", "不明")
                weekday = result.metadata.get("weekday_jp", "")
                content = result.page_content[:100].replace("\n", " ") + "..."

                print(f"  {j}. {event_date}({weekday})")
                print(f"     {content}")
        else:
            print("  📭 現在日時以降の今週の予定は見つかりませんでした")

        print()

        # テストクエリ
        test_queries = [
            "今週の予定を教えて",
            "今週何がある？",
            "今週のスケジュール",
            "週の予定",
            "今週何か予定ある？",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"🔍 テスト {i}: 「{query}」")
            print("-" * 40)

            # 今週の予定検索
            weekly_results = improver.search_weekly_schedule(query)

            print(f"📊 検索結果: {len(weekly_results)}件")

            if weekly_results:
                for j, result in enumerate(weekly_results[:3], 1):
                    event_date = result.metadata.get("event_date", "不明")
                    weekday = result.metadata.get("weekday_jp", "")
                    content = result.page_content[:100].replace("\n", " ") + "..."

                    print(f"  {j}. {event_date}({weekday})")
                    print(f"     {content}")
            else:
                print("  📭 今週の予定は見つかりませんでした")

            print()

        # 通常のschedule_aware_searchもテスト
        print("🧠 schedule_aware_search テスト")
        print("-" * 40)

        schedule_results = improver.schedule_aware_search("今週の予定を教えて", k=5)
        print(f"📊 schedule_aware_search結果: {len(schedule_results)}件")

        if schedule_results:
            for j, result in enumerate(schedule_results[:3], 1):
                content = result.page_content[:80].replace("\n", " ") + "..."
                print(f"  {j}. {content}")

        print()
        print("✅ 今週の予定検索機能テスト完了")

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("必要なパッケージがインストールされていない可能性があります")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_weekly_schedule()
