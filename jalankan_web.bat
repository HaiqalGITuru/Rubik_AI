@echo off
title Rubik AI
cd /d "%~dp0"
call .venv\Scripts\activate.bat

rem Pakai model besar (hasil Colab) kalau ada, selain itu model hasil laptop.
set MODEL=models\cubenet.pt
if exist models\cubenet_large_infer.pt set MODEL=models\cubenet_large_infer.pt

python -m web.server --model %MODEL%

pause
