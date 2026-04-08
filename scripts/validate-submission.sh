#!/usr/bin/env bash

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: ./validate-submission.sh <ping_url> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"

echo "========================================"
echo " OpenEnv Submission Validator"
echo "========================================"

# STEP 1: Check HF Space
echo "Step 1: Checking /reset endpoint..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$PING_URL/reset")

if [ "$HTTP_CODE" = "200" ]; then
  echo "PASSED -- HF Space is live"
else
  echo "FAILED -- HF Space not working"
  exit 1
fi

# STEP 2: Docker build
echo "Step 2: Building Docker..."

docker build "$REPO_DIR"

if [ $? -eq 0 ]; then
  echo "PASSED -- Docker build succeeded"
else
  echo "FAILED -- Docker build failed"
  exit 1
fi

# STEP 3: openenv validate
echo "Step 3: Running openenv validate..."

openenv validate

if [ $? -eq 0 ]; then
  echo "PASSED -- openenv validate passed"
else
  echo "FAILED -- openenv validate failed"
  exit 1
fi

echo "========================================"
echo " ALL CHECKS PASSED"
echo "========================================"
