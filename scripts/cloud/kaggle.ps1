# Shrink, upload and transcribe on Kaggle, start to finish. Each step resumes where it left off.
# Runs in its own window so it survives closing the Claude app, and asks Windows not to idle-
# sleep while it runs (closing the lid still sleeps). Log: data\cloud\kaggle.log
#
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\cloud\kaggle.ps1'

Set-Location (Join-Path $PSScriptRoot '..\..')
$env:PYTHONUTF8 = '1'
$py = '.\.venv\Scripts\python.exe'
$log = 'data\cloud\kaggle.log'

function Say($line) {
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep until this process exits.
Add-Type -Namespace Coach -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[Coach.Power]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null

foreach ($step in @('prepare', 'upload', 'run')) {
    Say "=== coach.cloud.kaggle $step ($(Get-Date))"
    & $py -m coach.cloud.kaggle $step 2>&1 | ForEach-Object { Say "$_" }
    if ($LASTEXITCODE -ne 0) {
        Say "=== $step failed (exit $LASTEXITCODE); fix it and rerun this script"
        break
    }
}
Say "=== finished ($(Get-Date))"
Read-Host 'Press Enter to close'
