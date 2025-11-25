# QA policy artifact bundle

This directory stores versioned QA policy bundles that can be uploaded to the ADK
ArtifactService. Each version folder contains the canonical YAML policy,
matching JSON Schema, and a manifest enumerating the files along with their
SHA-256 digests. A ready-to-upload tarball is produced alongside the folders.

## Generate / refresh bundles

```bash
PYTHONPATH=src python scripts/package_qa_policy.py --version v1
```

Outputs:
- `artifacts/qa_policy/v1/qa_policy.yaml`
- `artifacts/qa_policy/v1/qa_policy.schema.json`
- `artifacts/qa_policy/v1/manifest.json`
- `artifacts/qa_policy/qa_policy_v1.tar.gz`

## Upload to ADK ArtifactService

1. Run the packaging command above.
2. Authenticate with ADK (`adk auth login`).
3. Upload the tarball:
   ```bash
   adk artifacts upload \
     --id sparkle-motion/qa_policy \
     --version v1 \
     --file artifacts/qa_policy/qa_policy_v1.tar.gz
   ```
4. Record the resulting URI `artifact://sparkle-motion/qa_policy/v1` in docs and
   WorkflowAgent configs.

Once uploaded, the QA stage loads `qa_policy.yaml` by downloading and unpacking
that artifact. The schema file is included so the runner or QA adapter can
validate policy edits before enforcement.
