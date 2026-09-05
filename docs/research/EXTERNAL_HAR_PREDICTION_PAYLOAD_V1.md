# FoG metric-prediction payload, version 1

Status: locally retained, validated development evidence; not a published
release. The repository intentionally forbids `.npz` files in Git. The three
files below are required in addition to a clone. They contain predictions,
scoring labels and pseudonymous window/participant identities, not raw sensor
arrays or model weights. The pinned export manifest binds their exact bytes.
FoG-derived contents retain CC BY 4.0 and the attribution in the checkpoint's
`DATA_LICENSE_AND_SCOPE.md`; Apache-2.0 applies only to repository code.

| Payload-relative file | Bytes | SHA-256 | Evidence status |
|---|---:|---|---|
| fog_classical_observable_context/predictions.npz | 862218 | 09140171d14aedbc168feb0a5de9bc3159f087d80dead30112131addc5fffe1b | Validated development |
| fog_neural_nocudnn/predictions.npz | 137766 | de3a0f141b41d4954793b28ff6663d432465d462b0488a673426826154b4306b | Validated development |
| fog_posture_forest_negative/predictions.npz | 610250 | 47908adeb25ca1da33555d5b566444b3de7098251113526c6aeff5bbfe789471 | Validated negative candidate result |

Do not substitute another run's predictions, regenerate a favorable seed or
rename a sensor archive as a prediction file. Binary payload publication needs
separate authorization. Until it is supplied, reproduction from Git alone is
blocked by missing inputs, not by a missing model-training step.

## Restore into a fresh clone

Run this from the repository root in PowerShell. Supply the directory containing
the three payload subdirectories above. All inputs are checked before any copy;
existing destinations are refused. The table runner independently revalidates
the copied files against the result records. Keep the supplied attribution
document with any copy or redistribution of this payload.

```powershell
$payloadRoot = Read-Host 'Directory containing the separately supplied FoG prediction payload'
$evidenceRoot = 'results/research/cross_dataset_har_v3/observable_context_checkpoint_20260905'
$expected = @{
  'fog_classical_observable_context' = '09140171d14aedbc168feb0a5de9bc3159f087d80dead30112131addc5fffe1b'
  'fog_neural_nocudnn' = 'de3a0f141b41d4954793b28ff6663d432465d462b0488a673426826154b4306b'
  'fog_posture_forest_negative' = '47908adeb25ca1da33555d5b566444b3de7098251113526c6aeff5bbfe789471'
}
$copies = foreach ($name in $expected.Keys) {
  $source = Join-Path $payloadRoot "$name/predictions.npz"
  $destination = Join-Path $evidenceRoot "$name/predictions.npz"
  if (Test-Path -LiteralPath $destination) { throw "Refusing existing evidence: $destination" }
  if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$name]) {
    throw "Prediction SHA-256 mismatch: $source"
  }
  [PSCustomObject]@{ Source = $source; Destination = $destination }
}
foreach ($copy in $copies) {
  Copy-Item -LiteralPath $copy.Source -Destination $copy.Destination -ErrorAction Stop
}
```

Then run the locked-environment and `publication_table` commands in the README.
This is metric reconstruction with external, hash-bound inputs. It is not raw
data re-acquisition, model retraining, independent replication or confirmation.
