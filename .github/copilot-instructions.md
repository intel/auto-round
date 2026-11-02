When performing a code review, verify that any changes to markdown files (*.md) have corresponding updates in their Chinese translation files (those with _CN suffix, e.g., README.md → README_CN.md, GUIDE.md → GUIDE_CN.md). Ensure equivalent content and structure are maintained between the original and Chinese translation files.

## PyCharm and remote testing

- When the user works in PyCharm, prefer their configured remote environment for code validation and tests when one is available and accessible through the agent's tools. Do not assume that using PyCharm implies a remote environment exists, or that its saved connections and credentials are automatically accessible to the agent.
- Use the user's remote project directory and Python interpreter. Obtain missing connection details only when needed; never hard-code personal host aliases, usernames, project paths, or key paths in shared repository instructions.
- Before testing, verify that the remote code matches the intended local changes. Do not silently switch branches, overwrite remote changes, or assume PyCharm has synchronized files.
- Respect the user's GPU selection; if none is specified, check availability before GPU-intensive tests.
- If the preferred remote environment is inaccessible, report the blocker and confirm an alternative rather than silently falling back to local execution. Never store passwords or private-key contents in the repository.

