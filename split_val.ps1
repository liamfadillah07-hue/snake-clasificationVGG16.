# Split 15% data train -> val, per kelas
$ValSplit = 0.15
$Classes = @("venomous", "non_venomous")

foreach ($class in $Classes) {
    $trainDir = "data\train\$class"
    $valDir   = "data\val\$class"

    New-Item -ItemType Directory -Force -Path $valDir | Out-Null

    $allFiles = Get-ChildItem $trainDir -File | Get-Random -Count ([int]::MaxValue)
    $total = $allFiles.Count
    $valCount = [math]::Floor($total * $ValSplit)
    $valFiles = $allFiles[0..($valCount - 1)]

    foreach ($f in $valFiles) {
        Move-Item $f.FullName -Destination $valDir -Force
    }

    Write-Host "$class -> dipindahkan $($valFiles.Count) dari $total gambar ke val" -ForegroundColor Green
}

Write-Host ""
Write-Host "Ringkasan akhir:" -ForegroundColor Cyan
Get-ChildItem data -Recurse -Directory | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue).Count
    if ($n -gt 0) { Write-Host "$($_.FullName): $n gambar" }
}
