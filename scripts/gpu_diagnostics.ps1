# GPU diagnostics sampler for Windows
# Usage: run from PowerShell (as user):
#   .\gpu_diagnostics.ps1 -DurationSeconds 10 -OutFile gpu_diag.txt
param(
    [int]$DurationSeconds = 10,
    [int]$SampleInterval = 1,
    [string]$OutFile = "gpu_diag.txt"
)

Write-Output "Sampling GPU engine counters for $DurationSeconds seconds (interval=$SampleInterval s)" | Tee-Object -FilePath $OutFile
Write-Output "Timestamp;CounterPath;CookedValue" | Tee-Object -FilePath $OutFile -Append

# Counters to sample. These are generic GPU Engine counters present on Windows 10/11.
$counters = @(
    '\GPU Engine(* )\Utilization Percentage',
    '\GPU Engine(* )\Dedicated Usage (MB)'
)

# Sample counters repeatedly and append to out file
$iterations = [math]::Ceiling($DurationSeconds / $SampleInterval)
for ($i = 0; $i -lt $iterations; $i++) {
    $ts = Get-Date -Format o
    try {
        $results = Get-Counter -Counter $counters
        foreach ($sample in $results.CounterSamples) {
            $path = $sample.Path
            $val = $sample.CookedValue
            "${ts};${path};${val}" | Tee-Object -FilePath $OutFile -Append
        }
    } catch {
        "${ts};ERROR;$_" | Tee-Object -FilePath $OutFile -Append
    }
    Start-Sleep -Seconds $SampleInterval
}

Write-Output "Top processes by GPU memory shown in Task Manager (please run Task Manager -> Details -> add GPU Memory column)" | Tee-Object -FilePath $OutFile -Append
Write-Output "Process snapshot (Name,Id,WS) - WS is working set (system RAM), not GPU memory." | Tee-Object -FilePath $OutFile -Append
Get-Process | Sort-Object -Property WS -Descending | Select-Object -First 30 Name,Id,@{Name='WS_MB';Expression={[math]::Round($_.WS/1MB,2)}} | Format-Table | Out-String | Tee-Object -FilePath $OutFile -Append

Write-Output "Done sampling. Attach $OutFile and Task Manager process view for analysis." | Tee-Object -FilePath $OutFile -Append
