import whisper
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv(override=False)

class AudioIngestion():
    """Custom exception for audio ingestion errors."""

    def __init__(self, name: str):
        self.model = whisper.load_model(name)
    
    @traceable(name="Transcribe Audio", run_type="tool", save_result=True, use_cache=True)
    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribes an audio file using the Whisper model.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            str: The transcribed text from the audio file.
        """

        result = self.model.transcribe(file_path)
        return result["segments"]
