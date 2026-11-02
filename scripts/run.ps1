# Run one pipeline step in its own window, logging to data\<step>.log. A rerun picks up where
# the last one stopped (finished episodes are skipped).
#
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\run.ps1','correct','--all' -WindowStyle Minimized

param([Parameter(Mandatory)][string]$Step, [Parameter(ValueFromRemainingArguments)][string[]]$Rest)

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
$log = "data\$Step.log"

function Say($line) {
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep until this process exits.
Add-Type -Namespace Coach -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[Coach.Power]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null

Say "=== coach.$Step $Rest ($(Get-Date))"
# cmd keeps python's stderr as plain lines instead of PowerShell error records
cmd /c ".\.venv\Scripts\python.exe -m coach.$Step $Rest 2>&1" | ForEach-Object { Say "$_" }
Say "=== finished, exit $LASTEXITCODE ($(Get-Date))"
Read-Host 'Press Enter to close'
