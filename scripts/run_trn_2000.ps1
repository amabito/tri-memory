$py = "C:\Users\amabito\AppData\Local\Programs\Python\Python311\python.exe"
$script = "D:\work\Projects\trn\scripts\train_lm_realdata.py"

Set-Location "D:\work\Projects\trn\scripts"

Write-Host "=== TRN small 2000 steps ==="
& $py $script --model trn --size small --steps 2000 --batch-size 8 --seq-len 256 --log-every 500
Write-Host "TRN exit: $LastExitCode"

Write-Host ""
Write-Host "=== TF small 2000 steps ==="
& $py $script --model tf --size small --steps 2000 --batch-size 8 --seq-len 256 --log-every 500
Write-Host "TF exit: $LastExitCode"
