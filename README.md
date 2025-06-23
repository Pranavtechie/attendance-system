# Attendance System

## Setting up

1. Ensure you have `uv` installed on your system. If not download it from their official [GitHub Repo](https://github.com/astral-sh/uv)

2. Next, clone this repo locally, then open this repo and run the command `uv sync`. This will properly setup the `.venv` (Virtual Environment) folder on the current runtime that's used on this package, which is `3.10.17`

3. To start using the application you have to start two different processes, The flask server and the Qt UI process.

   - To start the flask server you can run the command

   ```bash
   gunicorn --workers 1 --bind 0.0.0.0:1337 src.api.index:app
   ```

   - To start the Qt UI you can run the command

   ```bash
   uv run attendance-app
   ```
