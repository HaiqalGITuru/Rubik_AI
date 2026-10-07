@echo off
title Rubik AI
cd /d "%~dp0"
call .venv\Scripts\activate.bat

rem Pakai model 3x3 besar (hasil Colab) kalau ada, selain itu model kecil hasil laptop.
rem Model 4x4 (models\cubenet_4x4.pt) otomatis dimuat oleh server kalau ada.
if exist models\cubenet_large_infer.pt (
    python -m web.server --model models\cubenet_large_infer.pt
) else (
    python -m web.server
)

pause
