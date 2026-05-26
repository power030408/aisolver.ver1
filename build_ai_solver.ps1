$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Entry = Join-Path $Root 'ai_solver.py'
$Dist = Join-Path $Root 'dist'
$Build = Join-Path $Root 'build'
$Spec = Join-Path $Root 'AISolver.spec'
$VendorTesseract = Join-Path $Root 'vendor\tesseract'

if (Test-Path $Dist) {
    Remove-Item -LiteralPath $Dist -Recurse -Force
}
if (Test-Path $Build) {
    Remove-Item -LiteralPath $Build -Recurse -Force
}
if (Test-Path $Spec) {
    Remove-Item -LiteralPath $Spec -Force
}

$Args = @(
  '--noconfirm',
  '--clean',
  '--windowed',
  '--name', 'AISolver',
  '--collect-all', 'google.genai',
  '--exclude-module', 'torch',
  '--exclude-module', 'torchvision',
  '--exclude-module', 'tensorflow',
  '--exclude-module', 'keras',
  '--exclude-module', 'pandas',
  '--exclude-module', 'scipy',
  '--exclude-module', 'matplotlib',
  '--exclude-module', 'onnxruntime',
  '--exclude-module', 'sqlalchemy'
)

if (Test-Path $VendorTesseract) {
    $Args += @('--add-data', "$VendorTesseract;vendor/tesseract")
}

$Args += $Entry
pyinstaller @Args

Write-Host "Build complete: $Dist\\AISolver"
