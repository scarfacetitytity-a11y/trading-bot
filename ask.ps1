# ask.ps1 — global AI model query shortcut
# Usage: .\ask.ps1 "your question"
#        .\ask.ps1 g: "groq only question"
#        .\ask.ps1 --repl
#        .\ask.ps1 --list
#
# Add to PowerShell profile for global 'ask' command:
#   Add-Content $PROFILE "`nfunction ask { python 'C:\Users\anton\Documents\trading-bot\execution\llm_advisor.py' @args }"

$BOT_DIR = "C:\Users\anton\Documents\trading-bot"
Push-Location $BOT_DIR
try {
    python -m execution.llm_advisor @args
} finally {
    Pop-Location
}
