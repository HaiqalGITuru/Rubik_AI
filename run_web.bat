@echo off
title Rubik AI
cd /d "%~dp0"
call .venv\Scripts\activate.bat

rem Use the large model (trained on Colab) if it exists, otherwise the laptop model.
set MODEL=models\cubenet.pt
if exist models\cubenet_large_infer.pt set MODEL=models\cubenet_large_infer.pt

python -m web.server --model %MODEL%

pause
