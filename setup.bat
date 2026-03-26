Installing uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

@echo off
echo Initializing project...
uv init

echo Adding dependencies...
uv add pandas
uv add pandas
uv add matplotlib
uv add seaborn
uv add ipykernel
uv add scikit-learn

echo Syncing environment...
uv sync

echo Done!
pause

code .
