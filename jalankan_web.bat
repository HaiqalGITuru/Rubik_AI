@echo off
title Rubik AI
cd /d "%~dp0"
call .venv\Scripts\activate.bat

rem Pakai model besar (hasil Colab/Kaggle) kalau ada, selain itu model hasil laptop.
set MODEL3=models\cubenet.pt
set MODEL4=models\cubenet_4x4.pt
if exist models\cubenet_large_infer.pt set MODEL3=models\cubenet_large_infer.pt
if exist models\cubenet_4x4_large_infer.pt set MODEL4=models\cubenet_4x4_large_infer.pt

python -m web.server --model %MODEL3% --model4 %MODEL4%

pause