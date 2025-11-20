<#
.SYNOPSIS
    Runs a series of Python scripts and measures their average execution time.
#>

# ----------------------------------------------------------------------
# 1. CONFIGURATION: Please modify this section
# ----------------------------------------------------------------------

# 配置参数
$path = ".\data\0925\"           # 影片目录路径
$f = "--force"                 # 强制重新执行标志

# (A) Set the Python commands to test
$commandsToTest = @(
    "python .\extract_frame_cache.py --video '$path' $f", 
    "python .\stage_pattern_analysis.py --video '$path' $f",
    "python .\auto_detect_machine_type.py --video '$path' $f",
    "python .\extract_roi_images.py --video '$path' $f",
    "python .\surgery_analysis_process.py --video '$path'$f"
)
# (B) Set the number of runs for each command
$numberOfRuns = 1


# ----------------------------------------------------------------------
# 2. EXECUTION & TIMING
# ----------------------------------------------------------------------

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$results = @()

Write-Host "🚀 Starting benchmark..."
Write-Host "   Testing $($commandsToTest.Count) scripts, $numberOfRuns runs each."
Write-Host ""

foreach ($command in $commandsToTest) {
    
    Write-Host "--------------------------------"
    Write-Host "📊 Testing: $command"
    Write-Host "--------------------------------"
    
    $runTimesSec = @()

    for ($i = 1; $i -le $numberOfRuns; $i++) {
        
        Write-Host "  ➡️ Run $i / $numberOfRuns..."
        Write-Host ""  # 换行，让 Python 输出显示在新行
        
        $startTime = Get-Date
        
        try {
            # 执行命令并捕获退出码
            Invoke-Expression $command
            $exitCode = $LASTEXITCODE
            
            $endTime = Get-Date
            $seconds = ($endTime - $startTime).TotalSeconds
            
            # 检查退出码
            if ($exitCode -eq 0 -or $null -eq $exitCode) {
                $runTimesSec += $seconds
                Write-Host ""
                Write-Host "  ✅ Done ( $("{0:N3}" -f $seconds) s )" -ForegroundColor Green
            } else {
                Write-Host ""
                Write-Host "  ❌ FAILED (exit code: $exitCode, took $("{0:N3}" -f $seconds) s)" -ForegroundColor Red
            }
            
        } catch {
            $endTime = Get-Date
            $seconds = ($endTime - $startTime).TotalSeconds
            Write-Host ""
            Write-Host "  ❌ FAILED (exception after $("{0:N3}" -f $seconds) s)" -ForegroundColor Red
            Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    if ($runTimesSec.Count -gt 0) {
        $stats = $runTimesSec | Measure-Object -Average -Minimum -Maximum
        
        $averageTime = $stats.Average
        $minTime = $stats.Minimum
        $maxTime = $stats.Maximum
        
        Write-Host ""
        Write-Host "  ✅ [ $command ]"
        Write-Host "     Average: $("{0:N3}" -f $averageTime) s"
        Write-Host "     Min: $("{0:N3}" -f $minTime) s, Max: $("{0:N3}" -f $maxTime) s" -ForegroundColor Gray
        Write-Host ""
        
        $results += [PSCustomObject]@{
            Command     = $command
            Status      = "成功"
            Runs        = $runTimesSec.Count
            AverageSecs = $averageTime
            MinSecs     = $minTime
            MaxSecs     = $maxTime
        }
    } else {
         Write-Host ""
         Write-Host "  ⚠️ [$command] 所有執行都失敗" -ForegroundColor Yellow
         Write-Host ""
         
         $results += [PSCustomObject]@{
            Command     = $command
            Status      = "❌ 失敗"
            Runs        = 0
            AverageSecs = "N/A"
            MinSecs     = "N/A"
            MaxSecs     = "N/A"
        }
    }
}


# ----------------------------------------------------------------------
# 3. FINAL REPORT
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "=============== ⏱️ FINAL REPORT ================" -ForegroundColor Cyan
$results | Format-Table -AutoSize -Wrap

# (Optional) Uncomment the line below to export the report to a CSV file
# $results | Export-Csv -Path "./python_timing_report.csv" -NoTypeInformation -Encoding UTF8
# Write-Host "Report saved to ./python_timing_report.csv"

Write-Host "=============== BENCHMARK COMPLETE ================" -ForegroundColor Cyan