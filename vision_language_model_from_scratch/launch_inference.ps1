param(
  [string]$CondaEnv = "d2l-zh",
  [string]$ModelPath = "",
  [string]$Prompt = "what is this picture taking about",
  [string]$ImageFilePath = "",
  [int]$MaxTokensToGenerate = 100,
  [double]$Temperature = 0.8,
  [double]$TopP = 0.9,
  [ValidateSet("True", "False")]
  [string]$DoSample = "False",
  [ValidateSet("True", "False")]
  [string]$OnlyCpu = "False"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
  $ModelPath = Join-Path $ScriptDir "3b-pt"
}

if ([string]::IsNullOrWhiteSpace($ImageFilePath)) {
  $ImageFilePath = Join-Path $ScriptDir "test_images\sea.jpg"
}

conda run --no-capture-output -n $CondaEnv python (Join-Path $ScriptDir "inference.py") `
  --model_path $ModelPath `
  --prompt $Prompt `
  --image_file_path $ImageFilePath `
  --max_tokens_to_generate $MaxTokensToGenerate `
  --temperature $Temperature `
  --top_p $TopP `
  --do_sample $DoSample `
  --only_cpu $OnlyCpu

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
