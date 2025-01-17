# verbatim

This is a commandline tool for creating transcripts of conversations recorded
as audio files. It uses the following AI models to achieve that:

- [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) for
    indentifying speakers
- [`OpenAI Whisper`](https://github.com/openai/whisper) for transcribing
    what the speakers say into text

## Requirements

### ffmpeg installed

Download [ffmpeg](https://ffmpeg.org/download.html) and install it.

### Hugging face token

The script requires a token to the huggingface API for downloading
`pyannote.audio` models.
[Here's](https://huggingface.co/docs/hub/security-tokens) how do get one.

### Accepting terms and conditions

In order to download the `pyannote.audio` models you need to accept their terms
and conditions. More on that [here](https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr).

## How to run

### Get the code

Clone this repository using git:

```bash
git clone https://github.com/jannawro/verbatim.git
```

### Installing dependencies

```bash
cd verbatim
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run

Run:

```bash
python verbatim/main.py \
    --audio-file sample.mp3 \
    --audio-format mp3 \
    --hugging-face-token hf_1234567890 \
    --speakers 2 \
    --output transcript.txt
```

To see all options run:

```bash
python verbatim/main.py --help
```

## Recommendations

### Use the optimal --whisper-model

Use whisper models variants according to [recommendations from OpenAI](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages).
This can be set via the `--whisper-model` flag.
From initial testing models smaller than "large" work fine for english. For
satysfying results with other languages "large" is recommended.

### Use the optimal number of --workers

Verbatim will spin up the amount of threads specified by `--workers`. This
number cannot be greater than the number of your CPU cores. Please note that
each worker will also spin up a separate instance of Whisper for parallel
processing. Use carefully togheter with `--whisper-model` to make sure you
have enough resources to spin up the number of Whispers of designated size.
