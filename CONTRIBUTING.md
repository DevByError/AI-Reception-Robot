# Contributing

Bug reports and pull requests are welcome.

## Reporting a bug

Open an issue with what you did, what you expected and what happened. Include the startup log
(it lists the voice, speech recognizer, devices and models that were found), your OS and Python
version, and whether you used `requirements.txt` or `requirements-lite.txt`. Remove API keys,
passwords and personal data from logs before posting them.

## Making a change

1. Fork the repository and create a branch from `main`.
2. Set up the environment as described in [docs/SETUP.md](docs/SETUP.md) and install the dev
   tools with `pip install -r requirements-dev.txt`.
3. Keep the change focused. If it adds a setting, add it to `.env.example` with its default.
4. Format and lint, then run the tests:

   ```bash
   black .
   ruff check .
   python -m pytest -m "not e2e"
   ```

5. Open a pull request that explains what changed and why.

## Privacy

Never commit face images, embeddings, voice recordings, conversation logs, database files or
`.env`. The `.gitignore` covers the usual locations; check `git status` before committing.
