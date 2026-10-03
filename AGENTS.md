# Minecraft Hand Detector

Commit every completed change or improvement after appropriate verification.
Keep commits focused and describe the resulting behavior. Never commit local
camera settings, downloaded models, virtual environments, or temporary files.

Use Kimi agents for implementation. Use GPT-6.1 Sol with High reasoning and
Fast mode when available for review checks. Run independent work in parallel
when possible. Preserve the user's requested gesture mapping and input safety.

Validate with `.venv\Scripts\python.exe -m unittest discover -s tests -t .`.
Synthetic tests do not establish real camera-to-Minecraft reliability.
