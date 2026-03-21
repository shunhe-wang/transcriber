# Meeting Transcriber — Setup & Usage

A free, local, private meeting transcriber for macOS.  
Records audio (mic or system audio) and transcribes using **OpenAI Whisper** — entirely on your Mac, no API keys, no internet required after setup.

---

## One-Time Setup (≈5 minutes)

### 1. Install Python (if not already installed)
```bash
python3 --version   # Should be 3.8+
```
If not installed: https://www.python.org/downloads/

### 2. Install Python packages
```bash
pip3 install openai-whisper sounddevice numpy
```
> If `pip3` isn't found, try `pip` instead.

### 3. Install BlackHole (to capture Zoom / system audio)
BlackHole is a free virtual audio driver that lets apps capture what's playing through your speakers.

```bash
brew install blackhole-2ch
```
Or download from: https://github.com/ExistingUserAccount/BlackHole

**After installing BlackHole:**
1. Open **Audio MIDI Setup** (search in Spotlight)
2. Click **+** → **Create Multi-Output Device**
3. Check both **BlackHole 2ch** and your normal speakers/headphones
4. Name it something like "Zoom + BlackHole"
5. In **System Preferences → Sound → Output**, select "Zoom + BlackHole"
6. In **Zoom → Settings → Audio → Speaker**, select "Zoom + BlackHole"

Now Zoom audio plays through your speakers AND gets captured by BlackHole.

---

## Running the App

```bash
python3 transcriber.py
```

---

## Usage

1. **Select input device** — choose "BlackHole 2ch" for Zoom/system audio, or your mic
2. **Select Whisper model:**
   - `tiny` — fastest, less accurate
   - `base` — good balance (recommended to start)
   - `small` / `medium` — more accurate, slower
   - `large` — most accurate, needs more RAM (~10GB), slowest
3. **Click "Start Recording"** — red timer counts up
4. **Click "Stop Recording"** when done
5. **Click "Transcribe"** — first run downloads the model (~140MB for base)
6. **Click "Save"** to export the transcript as a `.txt` file

---

## Model Download Sizes (one-time)

| Model  | Size   | Notes                        |
|--------|--------|------------------------------|
| tiny   | ~39 MB | Fast, lower accuracy         |
| base   | ~74 MB | Good starting point          |
| small  | ~244 MB| Better accuracy              |
| medium | ~769 MB| Great accuracy               |
| large  | ~1.5 GB| Best, needs 10GB RAM         |

Models are cached at `~/.cache/whisper/` after first download.

---

## Tips

- For a 1-hour meeting, `base` or `small` is usually fast enough and accurate
- Transcription of a 1-hour recording takes roughly 5–15 min on `base` depending on your Mac
- Apple Silicon Macs (M1/M2/M3) are significantly faster at transcription
- You can re-transcribe the same recording with a different model without re-recording

---

## Troubleshooting

**"No module named sounddevice"**  
→ Run: `pip3 install sounddevice`

**"No module named whisper"**  
→ Run: `pip3 install openai-whisper`

**No audio captured / silent recording**  
→ Make sure BlackHole is set as input device AND Zoom is outputting to your Multi-Output Device

**App won't open**  
→ Make sure you're running Python 3.8+: `python3 --version`
