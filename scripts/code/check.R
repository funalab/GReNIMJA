
# ファイル名
filename <- "./gene_gff/synonym"

# readLinesで読み込む（warningは無視しない）
lines <- readLines(filename, warn = TRUE)

# ファイルをバイナリとして読み込んで末尾チェック
raw <- readBin(filename, what = "raw", n = file.info(filename)$size)

# 改行コードが末尾にあるかどうかをチェック
ends_with_newline <- raw[length(raw)] %in% charToRaw("\n")

# 結果表示
if (!ends_with_newline) {
  cat("⚠️ 最終行が改行で終わっていません:\n")
  cat(lines[length(lines)], "\n")
} else {
  cat("✅ 最終行は改行で終わっています。\n")
}