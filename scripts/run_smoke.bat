@echo off
set PYTHON=C:\Users\amabito\AppData\Local\Programs\Python\Python311\python.exe
set SCRIPT=D:\work\Projects\trn\scripts\train_lm_realdata.py

echo === TRN small 2000 steps ===
%PYTHON% %SCRIPT% --model trn --size small --steps 2000 --batch-size 8 --seq-len 256 --log-every 200
echo TRN exit code: %ERRORLEVEL%

echo.
echo === TF small 2000 steps ===
%PYTHON% %SCRIPT% --model tf --size small --steps 2000 --batch-size 8 --seq-len 256 --log-every 200
echo TF exit code: %ERRORLEVEL%
