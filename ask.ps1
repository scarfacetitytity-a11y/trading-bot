# ask.ps1 — global AI model query shortcut
# Usage: .\ask.ps1 "your question"
#        .\ask.ps1 g: "groq only question"
#        .\ask.ps1 --repl
#        .\ask.ps1 --list
#
# Add to PowerShell profile for global 'ask' command:
#   Add-Content $PROFILE "`nfunction ask { & '<path-to-trading-bot>\ask.ps1' @args }"

$BOT_DIR = if ($env:AIDEN_BOT_DIR) { $env:AIDEN_BOT_DIR } else { $PSScriptRoot }
Push-Location $BOT_DIR
try {
    python -m execution.llm_advisor @args
} finally {
    Pop-Location
}
