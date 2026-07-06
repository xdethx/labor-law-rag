# scripts/smoke_m5.ps1 - M5 HTTP-level smoke test (PS 5.1 uyumlu)
$base = "http://localhost:8000"
$auth = @{ Authorization = "Bearer 12345" }   # kendi key'in
$pdf  = "data/raw/sample_contract.pdf"
$pass = 0; $fail = 0
function Check($name, $ok) {
    if ($ok) { Write-Host "PASS  $name" -ForegroundColor Green; $script:pass++ }
    else     { Write-Host "FAIL  $name" -ForegroundColor Red;   $script:fail++ }
}
# --- 0. Rerank-off retest: latency + score olcegi ---
$up = curl.exe -s -X POST "$base/contracts" -H "Authorization: Bearer 12345" -F "file=@$pdf" | ConvertFrom-Json
$sid = $up.session_id
$bytes = [Text.Encoding]::UTF8.GetBytes((@{ question = "sözleşmemde deneme süresi ne kadar, kanuna uygun mu?"; session_id = $sid } | ConvertTo-Json))
$t = Measure-Command {
    $resp = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base/ask" -Headers $auth -ContentType "application/json; charset=utf-8" -Body $bytes
}
$r = [Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray()) | ConvertFrom-Json
$maxScore = ($r.sources | Measure-Object -Property score -Maximum).Maximum
$minScore = ($r.sources | Measure-Object -Property score -Minimum).Minimum
Check "ask latency < 10s (rerank off)" ($t.TotalSeconds -lt 10)
Check "law scores rank-based (0..1, logit degil)" ($minScore -ge 0 -and $maxScore -le 1.5)
Check "answer cites both corpora" ($r.answer -match "Sözleşme" -and $r.answer -match "Madde")
Write-Host ("      latency={0:N1}s  law score range=[{1:N3}, {2:N3}]" -f $t.TotalSeconds, $minScore, $maxScore)
# --- Izolasyon ---
$bytes2 = [Text.Encoding]::UTF8.GetBytes((@{ question = "sözleşmemde deneme süresi ne kadar?"; session_id = "nonexistent-id" } | ConvertTo-Json))
$resp2 = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base/ask" -Headers $auth -ContentType "application/json; charset=utf-8" -Body $bytes2
$r2 = [Text.Encoding]::UTF8.GetString($resp2.RawContentStream.ToArray()) | ConvertFrom-Json
Check "isolation: wrong session sees no contract" ($r2.contract_sources.Count -eq 0)
# --- Hardening fixture'lari ---
"not a pdf" | Out-File fake.txt
Copy-Item fake.txt fake.pdf -Force
python -c "from pypdf import PdfReader, PdfWriter; r=PdfReader('$pdf'); w=PdfWriter(); [w.add_page(p) for p in r.pages]; w.encrypt('x'); w.write('encrypted.pdf')"
python -c "from pypdf import PdfReader, PdfWriter; r=PdfReader('$pdf'); w=PdfWriter(); [w.add_page(r.pages[0]) for _ in range(25)]; w.write('big.pdf')"
function UploadCode($file, $withAuth = $true) {
    if ($withAuth) { curl.exe -s -o NUL -w "%{http_code}" -X POST "$base/contracts" -H "Authorization: Bearer 12345" -F "file=@$file" }
    else           { curl.exe -s -o NUL -w "%{http_code}" -X POST "$base/contracts" -F "file=@$file" }
}
Check "non-PDF (.txt) rejected 4xx"        ((UploadCode "fake.txt")      -match "^(400|415)$")
Check "fake .pdf rejected 4xx"             ((UploadCode "fake.pdf")      -match "^(400|415)$")
Check "encrypted PDF rejected 400"         ((UploadCode "encrypted.pdf") -match "^(400|415)$")
Check "oversized PDF (25p) rejected 400"   ((UploadCode "big.pdf")       -match "^(400|413|415)$")
Check "no-auth upload rejected 401/403"    ((UploadCode $pdf $false)     -match "^(401|403)$")
# --- Delete + health ---
$del = curl.exe -s -o NUL -w "%{http_code}" -X DELETE "$base/contracts/$sid" -H "Authorization: Bearer 12345"
$h = Invoke-RestMethod -Uri "$base/health"
Check "delete returns 2xx"                 ($del -match "^2")
Check "contracts_points back to 0"         ($h.contracts_points -eq 0)
Remove-Item fake.txt, fake.pdf, encrypted.pdf, big.pdf -ErrorAction SilentlyContinue
Write-Host "`n$pass PASS / $fail FAIL"
