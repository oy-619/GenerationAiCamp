# GAS版へのデータインポート手順

## 📁 エクスポートされたファイル

### 1. conversation_history.csv
- **内容**: 会話履歴データ
- **インポート先**: Google Sheets の「会話履歴」シート
- **手順**:
  1. Google Sheets を開く
  2. 「ファイル」→「インポート」
  3. CSV ファイルをアップロード
  4. 既存データに追加

### 2. chroma_documents.csv
- **内容**: ドキュメントデータベース
- **インポート先**: Google Sheets の「ドキュメント」シート
- **注意**: GAS版では簡易検索のみ対応

### 3. system_stats.json
- **内容**: システム統計情報
- **用途**: 移行前後の比較用

## 🔄 GAS版での処理

### データインポート用スクリプト (GAS)
```javascript
function importConversationHistory() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('会話履歴');

  // Google Drive にアップロードした CSV ファイルを読み込み
  // const fileId = 'YOUR_CSV_FILE_ID';
  // const file = DriveApp.getFileById(fileId);
  // const csvData = file.getBlob().getDataAsString();

  // CSV データを解析してシートに書き込み
  // 詳細は deployment_guide.md を参照
}
```

## ✅ インポート後の確認事項

1. **データ件数の一致確認**
2. **文字化けチェック**
3. **GAS動作テスト**
4. **応答品質の比較**

## 📞 サポート

問題が発生した場合は、system_stats.json の情報と共にお知らせください。
