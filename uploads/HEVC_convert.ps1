

Get-ChildItem -Path * -Include *.mp4,*.mkv,*.mov,*.avi -File | ForEach-Object {
    # Call ffprobe and capture the output string
    $codec = (ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 $_.FullName).Trim()

    if ($codec -eq "hevc") {
        Write-Host "Processing $($_.Name) - Format: $codec"
        $status = "Needs Re-encode (HEVC -> H.264)"
        $reencodeDir = Join-Path $_.DirectoryName "reencode"
        New-Item -ItemType Directory -Path $reencodeDir -Force | Out-Null
        $outFile = Join-Path $reencodeDir "$($_.BaseName)_reencode.mp4"
        ffmpeg -i $_.FullName -c:v libx264 -c:a copy $outFile | Out-Null
    } 
    else {
        $status = "Already HEVC"
    }

    [PSCustomObject]@{
        FileName = $_.Name
        Format   = $codec
        Status   = $status
    }
} | Format-Table -AutoSize
