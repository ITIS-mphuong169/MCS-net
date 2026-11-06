@echo off
setlocal EnableExtensions
title CLIP WikiArt - single + detail, 100 epoch
cd /d "%~dp0.."
set "PYTHONUTF8=1"

echo ==============================================================
echo  CLIP WikiArt: 2 run (single, detail), moi run 100 epoch
echo  Bi dung giua chung: chay lai file nay, tu tiep tuc tu epoch cuoi
echo ==============================================================
echo.

rem ---- Python
set "PY=python"
%PY% --version >nul 2>nul
if errorlevel 1 set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 goto :nopython

rem ---- phai dang o thu muc MCS-Net co code moi nhat
if not exist "server\clip.py" goto :nocode

rem ---- duong dan thu muc WikiArt: hoi mot lan, nho lai
set "DATA="
if exist "%USERPROFILE%\.mcs_data_dir" set /p DATA=<"%USERPROFILE%\.mcs_data_dir"
if not defined DATA set /p DATA=Nhap duong dan thu muc WikiArt, vi du D:\wikiart : 
if not defined DATA goto :nodata
set "DATA=%DATA:"=%"
if not defined DATA goto :nodata
if "%DATA:~-1%"=="\" set "DATA=%DATA:~0,-1%"
if not exist "%DATA%\" goto :nodata
> "%USERPROFILE%\.mcs_data_dir" echo %DATA%
echo Du lieu: %DATA%
echo.

echo [1/6] Cai goi, kiem tra PyTorch dung duoc GPU
%PY% server\clip.py setup
if errorlevel 1 goto :fail
echo.

echo [2/6] Dang nhap W^&B
%PY% server\clip.py login
if errorlevel 1 goto :fail
echo.

echo [3/6] Kiem tra may
%PY% server\clip.py check --data "%DATA%"
if errorlevel 1 goto :fail
echo.

echo [4/6] Chay thu 10 phut, chi lan dau
%PY% server\clip.py train smoke --data "%DATA%"
if errorlevel 1 goto :fail
echo.

echo [5/6] Train 2 run: single roi detail, moi run 100 epoch
%PY% server\clip.py train both --data "%DATA%"
if errorlevel 1 goto :fail
echo.

echo [6/6] Cham test va McNemar
%PY% server\clip.py eval both --data "%DATA%"
if errorlevel 1 goto :fail
echo.

echo ==============================================================
echo  XONG. Gui toi file: %USERPROFILE%\results\summary_clip.txt
echo ==============================================================
pause
exit /b 0

:nopython
echo !! Khong thay Python. Cai Python 3.10 tro len tu python.org, tich "Add python.exe to PATH", roi chay lai file nay.
goto :fail

:nocode
echo !! Khong thay server\clip.py. Mo PowerShell trong thu muc MCS-Net va chay:  git pull origin exp/regularize
goto :fail

:nodata
echo !! Khong thay thu muc du lieu WikiArt. Xoa file %USERPROFILE%\.mcs_data_dir roi chay lai file nay de nhap duong dan khac.
goto :fail

:fail
echo.
echo !! DUNG vi loi o tren. Chup man hinh hoac copy cac dong do gui toi.
echo !! Sua xong thi chay lai file nay: se tiep tuc, khong lam lai tu dau.
pause
exit /b 1
