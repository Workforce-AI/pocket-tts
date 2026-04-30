#!/bin/bash
# cache-voices.sh
# Run once after each deploy to warm the voice cache.
# Usage: bash cache-voices.sh [API_URL]
# Default API_URL: http://localhost:8000

set -e

API_URL="${1:-http://localhost:8000}"
API_KEY=$(grep -E '^POCKET_TTS_API_KEY=' .env 2>/dev/null | cut -d= -f2-)

if [ -z "$API_KEY" ]; then
  echo "Warning: POCKET_TTS_API_KEY not found in .env — sending requests without auth"
fi

cache_voice() {
  local name="$1"
  local url="$2"

  echo -n "  Caching $name ... "
  local t0=$SECONDS

  response=$(curl -sf -X POST "$API_URL/voices/cache" \
    ${API_KEY:+-H "X-API-Key: $API_KEY"} \
    -F "name=$name" \
    -F "voice_url=$url" 2>&1) || {
    echo "FAILED"
    echo "    Error: $response"
    return 1
  }

  local elapsed=$(( SECONDS - t0 ))
  local voice_id
  voice_id=$(echo "$response" | grep -o '"voice_id":"[^"]*"' | cut -d'"' -f4)
  echo "done in ${elapsed}s  →  $voice_id"
}

echo ""
echo "Pocket TTS — Voice Cache Warmer"
echo "Target: $API_URL"
echo "────────────────────────────────────────"

cache_voice "default"   "https://cdn.hireworkforce.ai/voice/default.mp3"
cache_voice "jesus"     "https://cdn.hireworkforce.ai/voice/jesus.mp3"
cache_voice "lauren"    "https://cdn.hireworkforce.ai/voice/lauren.mp3"
cache_voice "martin"    "https://cdn.hireworkforce.ai/voice/martin.mp3"
cache_voice "isabella"  "https://cdn.hireworkforce.ai/voice/isabella.mp3"
cache_voice "andre"     "https://cdn.hireworkforce.ai/voice/andre.mp3"
cache_voice "vivian"    "https://cdn.hireworkforce.ai/voice/vivian.mp3"
cache_voice "adrian"    "https://cdn.hireworkforce.ai/voice/adrian.mp3"
cache_voice "bruno"     "https://cdn.hireworkforce.ai/voice/bruno.mp3"
cache_voice "mia"       "https://cdn.hireworkforce.ai/voice/mia.mp3"
cache_voice "eric"      "https://cdn.hireworkforce.ai/voice/eric.mp3"

echo "────────────────────────────────────────"
echo "All voices cached. Current cache:"
echo ""
curl -sf "$API_URL/voices" ${API_KEY:+-H "X-API-Key: $API_KEY"} \
  | grep -o '"voice_id":"[^"]*","name":"[^"]*"' \
  | sed 's/"voice_id":"\([^"]*\)","name":"\([^"]*\)"/  \2  →  \1/'
echo ""
