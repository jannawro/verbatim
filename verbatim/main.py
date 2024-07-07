from argparse import ArgumentParser
from verbatim import Verbatim
from verbatim import format_text, format_xml_like


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--audio-file", type=str, required=True, help="Audio file to be transcribed."
    )
    parser.add_argument(
        "--audio-format",
        type=str,
        required=True,
        help="File format of the audio file.",
    )
    parser.add_argument(
        "--format",
        default="text",
        type=str,
        required=False,
        help="Format of the output. Currently supported formats: 'text', or 'xml'. (Default is 'text')",
    )
    parser.add_argument(
        "--hugging-face-token",
        default="",
        type=str,
        required=True,
        help="Token for accessing https://huggingface.co/ API. More on hugging face security tokens: https://huggingface.co/docs/hub/security-tokens",
    )
    parser.add_argument(
        "--output",
        default="stdout",
        type=str,
        required=False,
        help="Output path for the transcription. Override to write the output to a file. (Default is 'stdout' - the output is printed to the console)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        type=str,
        required=False,
        help="Model size used for OpenAI Whisper. Learn more on different sizes: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages (Default is 'base')",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        required=True,
        help="Number of speakers in the audio file.",
    )

    args = parser.parse_args()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"    > {arg}: {value}")

    formatter = None
    match args.format:
        case "text":
            formatter = format_text
        case "xml":
            formatter = format_xml_like
        case _:
            raise Exception("Invalid formatter.")

    v = Verbatim(
        audio_file_path=args.audio_file,
        audio_format=args.audio_format,
        formatter=formatter,
        hugging_face_token=args.hugging_face_token,
        output_style=args.output,
        whisper_model=args.whisper_model,
        speakers=args.speakers,
    )

    try:
        v.run()

    except Exception as e:
        print(f"An error occured: {e}")

    finally:
        v.cleanup()


if __name__ == "__main__":
    main()
