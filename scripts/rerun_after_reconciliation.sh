#!/bin/sh
# Rebuild everything downstream of the acceptance-time correction (docs/research_protocol.md change log).
set -e
cd "$(dirname "$0")/.."
FS=.venv/bin/filingshock
$FS build --allow-placeholder-agent
.venv/bin/python -c "
from filingshock.config import Config; from filingshock.storage import DataLayout; from filingshock.audit import acceptance_reconciliation_summary
import json; l=DataLayout(Config.load().data_dir); s=acceptance_reconciliation_summary(l, l.processed/'events_pre_reconciliation.parquet')
print(json.dumps({k:v for k,v in s.items() if k!='previously_eligible_now_excluded'}, indent=1)); print('previously eligible now excluded:', s.get('previously_eligible_now_excluded'))"
$FS experiment --name primary --set-primary --note "Protocol v1.0 primary analysis on index-page acceptance times (corrected data)"
$FS sensitivity
$FS verify-artifacts
