param(
    [string]$PythonRuntime = "C:\Users\Xiaolei\.cache\codex-runtimes\codex-primary-runtime\dependencies\python",
    [string]$UserSitePackages = "C:\Users\Xiaolei\AppData\Roaming\Python\Python312\site-packages"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$destination = Join-Path $root "dist\PFAS_CCS_Screening"
$resolvedRoot = [System.IO.Path]::GetFullPath($root)
$resolvedDestination = [System.IO.Path]::GetFullPath($destination)

if (-not $resolvedDestination.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to build outside the project directory: $resolvedDestination"
}
if (-not (Test-Path -LiteralPath $PythonRuntime)) {
    throw "Python runtime was not found: $PythonRuntime"
}
if (-not (Test-Path -LiteralPath $UserSitePackages)) {
    throw "Python user site-packages was not found: $UserSitePackages"
}

if (Test-Path -LiteralPath $destination) {
    Remove-Item -LiteralPath $destination -Recurse -Force
}
New-Item -ItemType Directory -Path $destination | Out-Null

Copy-Item -Path (Join-Path $PythonRuntime "*") -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $destination "pythonw.exe") `
    -Destination (Join-Path $destination "PFAS_CCS_Screening.exe") -Force

$destinationSite = Join-Path $destination "Lib\site-packages"
$dependencyItems = @(
    "sklearn",
    "scikit_learn-1.9.0.dist-info",
    "scipy",
    "scipy.libs",
    "scipy-1.17.1.dist-info",
    "joblib",
    "joblib-1.5.3.dist-info",
    "threadpoolctl.py",
    "threadpoolctl-3.6.0.dist-info"
)
foreach ($item in $dependencyItems) {
    $source = Join-Path $UserSitePackages $item
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $destinationSite -Recurse -Force
    }
}

Copy-Item -LiteralPath (Join-Path $root "pfas_screening_app") `
    -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $root "models") `
    -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $root "packaging\sitecustomize.py") `
    -Destination $destination -Force
Copy-Item -LiteralPath (Join-Path $root "packaging\python312._pth") `
    -Destination $destination -Force
Copy-Item -LiteralPath (Join-Path $root "README_APP_zh-CN.md") `
    -Destination $destination -Force
Copy-Item -LiteralPath (Join-Path $root "README_APP.md") `
    -Destination $destination -Force
Copy-Item -LiteralPath (Join-Path $root "Chemical List PFAS.xlsx") `
    -Destination $destination -Force

Write-Host ""
Write-Host "Portable application created at:"
Write-Host (Join-Path $destination "PFAS_CCS_Screening.exe")
