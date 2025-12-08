<#
GPU process memory sampler
Usage (run as admin for best results):
  .\gpu_proc_sampler.ps1 -Duration 30 -OutFile ..\gpu_proc_usage.csv
#>
param(
    [int]$Duration = 30,
    [string]$OutFile = "..\gpu_proc_usage.csv",
    [int]$Interval = 1
)

Write-Output "Sampling GPU process memory for $Duration seconds -> $OutFile"
"Timestamp,Instance,PID,Process,GPU_MB" | Out-File -FilePath $OutFile -Encoding utf8

$iterations = [math]::Ceiling($Duration / $Interval)
for ($i = 0; $i -lt $iterations; $i++) {
    $samples = $null
    try {
        $samples = Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage' -SampleInterval 1 -MaxSamples 1
    } catch {
        # some systems expose different paths; try the MB-suffixed path
        try {
            $samples = Get-Counter -Counter "\\GPU Process Memory(*)\\Dedicated Usage (MB)" -SampleInterval 1 -MaxSamples 1
        } catch {
            Write-Output "Warning: counters unavailable for this sample"
        }
    }
    if ($samples) {
        foreach ($s in $samples.CounterSamples) {
            $inst = $s.InstanceName
            # Prefer explicit 'pid_####' anywhere in the instance string
            $procId = $null
            if ($inst -match 'pid_(\d{2,8})') { $procId = [int]$matches[1] }
            elseif ($inst -match '_(\d{2,8})$') { $procId = [int]$matches[1] }
            elseif ($inst -match '\((\d{2,8})\)$') { $procId = [int]$matches[1] }
            $procName = '<unknown>'
            if ($procId -ne $null -and $procId -gt 0) {
                $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($p) { $procName = $p.ProcessName }
            }
            $line = "$(Get-Date -Format o),$inst,$procId,$procName,$([math]::Round($s.CookedValue,2))"
            $line | Out-File -FilePath $OutFile -Append -Encoding utf8
        }
    }
    Start-Sleep -Seconds $Interval
}
Write-Output "Sampling complete -> $OutFile"
