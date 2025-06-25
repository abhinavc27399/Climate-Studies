# Get all subdirectories with a main.py file
$folders = Get-ChildItem -Directory | Where-Object {
    Test-Path "$($_.FullName)\main.py"
}

if ($folders.Count -eq 0) {
    Write-Host "❌ No subdirectories with main.py found."
    exit
}

# Display options
Write-Host "`nSelect which main.py you want to run:`n"
for ($i = 0; $i -lt $folders.Count; $i++) {
    Write-Host "$($i + 1)) $($folders[$i].Name)"
}

# Read user input
$choice = Read-Host "`nEnter a number"
$index = [int]$choice - 1

if ($index -ge 0 -and $index -lt $folders.Count) {
    $selectedFolder = $folders[$index].FullName
    Write-Host "`n▶️ Running: $selectedFolder\\main.py"
    python "$selectedFolder\main.py"
} else {
    Write-Host "❌ Invalid selection."
}