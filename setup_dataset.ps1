# ============================================================
# Script: setup_dataset.ps1
# Fungsi: Menyusun ulang dataset "Snake Images" (Kaggle) ke
#         struktur folder yang dibutuhkan oleh train_model.py:
#           data/train/venomous, data/train/non_venomous
#           data/val/venomous,   data/val/non_venomous
#           data/test/venomous,  data/test/non_venomous
#
# Cara pakai:
#   1. Simpan file ini di root folder proyek
#      (snake-classifier-flask-app\setup_dataset.ps1)
#   2. Sesuaikan variabel $SourceRoot di bawah jika lokasi
#      hasil ekstrak Kaggle berbeda.
#   3. Jalankan:  .\setup_dataset.ps1
# ============================================================

# --- KONFIGURASI: sesuaikan jika perlu ---
$SourceRoot = "$env:USERPROFILE\Downloads\archive (4)\Snake Images"
$DestRoot   = "$PSScriptRoot\data"
$ValSplit   = 0.15   # 15% dari data train dipindah jadi data validasi

# Nama folder sumber (case-insensitive, tidak peduli spasi/underscore)
$ClassMap = @{
    "venomous"     = "Venomous"
    "non_venomous" = "Non Venomous"
}

Write-Host "Sumber dataset : $SourceRoot"
Write-Host "Tujuan         : $DestRoot"
Write-Host ""

if (!(Test-Path $SourceRoot)) {
    Write-Host "ERROR: Folder sumber tidak ditemukan: $SourceRoot" -ForegroundColor Red
    Write-Host "Silakan cek ulang path-nya dan sesuaikan variabel `$SourceRoot di dalam script ini."
    exit 1
}

foreach ($classKey in $ClassMap.Keys) {
    $classFolderName = $ClassMap[$classKey]

    # ---------- 1. Salin data TEST ----------
    $testSrc = Join-Path $SourceRoot "test\$classFolderName"
    $testDst = Join-Path $DestRoot "test\$classKey"

    if (Test-Path $testSrc) {
        New-Item -ItemType Directory -Force -Path $testDst | Out-Null
        Copy-Item "$testSrc\*" -Destination $testDst -Force
        $count = (Get-ChildItem $testDst -File).Count
        Write-Host "[TEST]  $classKey -> $count gambar disalin" -ForegroundColor Green
    } else {
        Write-Host "[TEST]  Folder tidak ditemukan: $testSrc" -ForegroundColor Yellow
    }

    # ---------- 2. Split data TRAIN -> train + val ----------
    $trainSrc = Join-Path $SourceRoot "train\$classFolderName"
    $trainDst = Join-Path $DestRoot "train\$classKey"
    $valDst   = Join-Path $DestRoot "val\$classKey"

    if (Test-Path $trainSrc) {
        New-Item -ItemType Directory -Force -Path $trainDst | Out-Null
        New-Item -ItemType Directory -Force -Path $valDst   | Out-Null

        # Ambil semua file, acak urutannya
        $allFiles = Get-ChildItem $trainSrc -File | Get-Random -Count ([int]::MaxValue)
        $totalCount = $allFiles.Count
        $valCount = [math]::Floor($totalCount * $ValSplit)

        $valFiles   = $allFiles[0..($valCount - 1)]
        $trainFiles = $allFiles[$valCount..($totalCount - 1)]

        foreach ($f in $trainFiles) { Copy-Item $f.FullName -Destination $trainDst -Force }
        foreach ($f in $valFiles)   { Copy-Item $f.FullName -Destination $valDst -Force }

        Write-Host "[TRAIN] $classKey -> $($trainFiles.Count) gambar (train), $($valFiles.Count) gambar (val)" -ForegroundColor Green
    } else {
        Write-Host "[TRAIN] Folder tidak ditemukan: $trainSrc" -ForegroundColor Yellow
    }

    Write-Host ""
}

Write-Host "=============================================="
Write-Host "SELESAI. Ringkasan jumlah gambar per folder:"
Write-Host "=============================================="
Get-ChildItem $DestRoot -Recurse -Directory | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue).Count
    if ($n -gt 0) {
        Write-Host ("{0,-45} {1,5} gambar" -f $_.FullName.Replace($DestRoot, "data"), $n)
    }
}
