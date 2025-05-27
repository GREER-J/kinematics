$SourcePath = "src/dynamics_library"
$OutputDir = "out/docs"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Get-ChildItem -Recurse -Filter *.py -Path $SourcePath | ForEach-Object {
    $relativePath = $_.FullName -replace "\\", "/" -replace ".py$", ""
    $moduleName = $relativePath -replace "^.*?src/", "src/" -replace "/", "."
    Write-Host "Generating docs for $moduleName"
    python -m pydoc -w $moduleName
    Move-Item "$moduleName.html" $OutputDir -Force
}

Write-Host "All docs saved to $OutputDir/"
