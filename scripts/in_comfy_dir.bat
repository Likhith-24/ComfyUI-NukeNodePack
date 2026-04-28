@echo off
REM Activate the path-based comfy env (no `comfy` name env exists) and run any command.
call "C:\Users\varig\miniconda3\condabin\activate.bat" "D:\PROJECT\ComfyUI_windows_portable\comfy"
cd /d D:\PROJECT\ComfyUI_windows_portable\ComfyUI
%*
