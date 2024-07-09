from multiprocessing import Pool
import os
import shutil
from typing import Callable, List, TypedDict
import time

from halo import Halo
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
from whisper import Whisper, load_model


CHUNKS_TEMP_DIR = "./tmp_chunks"


class Chunk(TypedDict):
    speaker: str
    start: float
    end: float
    file: str
    text: str


def format_text(chunks: List[Chunk]) -> str:
    formatted_chunks = []
    for chunk in chunks:
        formatted_chunks.append(f"{chunk['speaker']}: {chunk['text']}")

    return "\n".join(formatted_chunks)


def format_xml_like(chunks: List[Chunk]) -> str:
    formatted_chunks = []
    for chunk in chunks:
        formatted_chunks.append(
            f"<{chunk['speaker']}>\n    <lines>{chunk['text']}</lines>\n</{chunk['speaker']}>"
        )

    return "\n".join(formatted_chunks)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def whisper(model_name: str, device: torch.device) -> Whisper:
    return load_model(model_name).to(device)


def transcribe_chunk(chunk: Chunk, whisper_model: str) -> Chunk:
    device = select_device()
    model = whisper(whisper_model, device)
    result = model.transcribe(chunk["file"])
    chunk["text"] = result["text"]
    return chunk


class Verbatim:
    def __init__(
        self,
        audio_file_path: str,
        audio_format: str,
        formatter: Callable[[List[Chunk]], str],
        hugging_face_token: str,
        output_style: str,
        whisper_model: str,
        speakers: int,
        workers: int,
    ) -> None:
        os.makedirs(CHUNKS_TEMP_DIR, exist_ok=True)
        self.audio_chunks_dir = CHUNKS_TEMP_DIR

        if audio_format != "wav":
            audio = AudioSegment.from_file(audio_file_path)
            conversion_output = f"{self.audio_chunks_dir}/converted.wav"
            audio.export(conversion_output, format="wav")
            self.audio_file = conversion_output
        else:
            self.audio_file = audio_file_path

        self.audio_format = audio_format
        self.formatter = formatter
        self.hugging_face_token = hugging_face_token
        self.output_style = output_style
        self.whisper_model = whisper_model
        self.speakers = speakers
        self.workers = workers

        return

    def whisper(self) -> Whisper:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return load_model(self.whisper_model).to(device)

    def find_chunks(self) -> List[Chunk]:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.hugging_face_token
        ).to(select_device())

        chunks: List[Chunk] = []

        start_time = time.time()
        spinner = Halo(text="Finding speakers and lines timestamps...", spinner="dots")
        spinner.start()

        try:
            diarization = pipeline(self.audio_file, num_speakers=self.speakers)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                chunks.append(
                    Chunk(
                        speaker=speaker,
                        start=turn.start,
                        end=turn.end,
                        file="",
                        text="",
                    )
                )
            duration = time.time() - start_time
            spinner.succeed(f"Finding chunks complete! (took {duration:.2f} seconds)")

        except Exception:
            spinner.fail(f"An error occured while finding chunks.")
            raise

        finally:
            spinner.stop()

        return chunks

    def squash_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []

        squashed_chunks: List[Chunk] = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if current_chunk["speaker"] == next_chunk["speaker"]:
                current_chunk = Chunk(
                    speaker=current_chunk["speaker"],
                    start=current_chunk["start"],
                    end=next_chunk["end"],
                    file="",
                    text="",
                )
            else:
                squashed_chunks.append(current_chunk)
                current_chunk = next_chunk

        squashed_chunks.append(current_chunk)

        return squashed_chunks

    def split_audio_by_chunk(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []

        audio = AudioSegment.from_file(self.audio_file)

        start_time = time.time()
        spinner = Halo(text="Splitting the audio file into chunks...", spinner="dots")
        spinner.start()

        try:
            for i, chunk in enumerate(chunks):
                start_ms = chunk["start"] * 1000
                end_ms = chunk["end"] * 1000

                segment = audio[start_ms:end_ms]

                output_file = f"{self.audio_chunks_dir}/chunk_{i+1}.wav"
                segment.export(output_file, format="wav")
                chunk["file"] = output_file
            duration = time.time() - start_time
            spinner.succeed(
                f"Splitting audio file finished! (took {duration:.2f} seconds)"
            )

        except Exception:
            spinner.fail(f"An error occured while splitting audio file chunks.")
            raise

        finally:
            spinner.stop()

        return chunks

    def transcribe_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []

        transcribed_chunks: List[Chunk] = []

        start_time = time.time()
        spinner = Halo(
            text="Transcribing the audio chunks into text...", spinner="dots"
        )
        spinner.start()

        try:
            with Pool(self.workers) as pool:
                transcribed_chunks = pool.starmap(
                    transcribe_chunk, [(chunk, self.whisper_model) for chunk in chunks]
                )

            duration = time.time() - start_time
            spinner.succeed(f"Transcribing succeeded! (took {duration:.2f} seconds)")

        except Exception:
            spinner.fail(f"An error occured while transcribing the audio.")
            raise

        finally:
            spinner.stop()

        return transcribed_chunks

    def format(self, chunks: List[Chunk]) -> str:
        if not chunks:
            return ""

        return self.formatter(chunks)

    def output(self, text: str) -> None:
        match self.output_style:
            case "stdout":
                print(text)
            case filename:
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(text)
        return

    def cleanup(self) -> None:
        shutil.rmtree(self.audio_chunks_dir)
        return

    def run(self) -> None:
        self.output(
            self.format(
                self.transcribe_chunks(
                    self.split_audio_by_chunk(self.squash_chunks(self.find_chunks()))
                )
            )
        )

        return
