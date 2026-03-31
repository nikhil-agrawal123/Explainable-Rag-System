"""
Tests for the document ingestion endpoints.

    POST /api/v3/ingest_pdf/    — PDF upload & chunking pipeline
    POST /api/v3/ingest_audio/  — Audio upload & transcription pipeline
"""
from unittest.mock import patch, AsyncMock


# =====================================================================
#  PDF ingestion
# =====================================================================

class TestIngestPDF:
    """Upload one or more PDF files for ingestion."""

    def test_single_pdf_upload(self, test_client, sample_pdf):
        """Uploading a valid PDF should return success."""
        with patch(
            "app.pipeline.stage_1_ingestion.IngestionPipeline.process_document",
            new_callable=AsyncMock,
            return_value={"status": "success", "doc_id": "test_upload", "chunks": 3},
        ):
            with open(sample_pdf, "rb") as f:
                resp = test_client.post(
                    "/api/v3/ingest_pdf/",
                    files=[("files", ("test_upload.pdf", f, "application/pdf"))],
                )
            assert resp.status_code == 200
            body = resp.json()
            assert "job_summary" in body
            assert body["job_summary"][0]["status"] == "success"

    def test_non_pdf_file_is_skipped(self, test_client, tmp_path):
        """A .txt file should be skipped with reason 'Not a PDF'."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("hello world")

        with open(txt_file, "rb") as f:
            resp = test_client.post(
                "/api/v3/ingest_pdf/",
                files=[("files", ("notes.txt", f, "text/plain"))],
            )
        assert resp.status_code == 200
        summary = resp.json()["job_summary"]
        assert summary[0]["status"] == "skipped"
        assert summary[0]["reason"] == "Not a PDF"

    def test_multiple_files_mixed(self, test_client, sample_pdf, tmp_path):
        """Mixing valid PDFs and invalid files should handle each correctly."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("not a pdf")

        with patch(
            "app.pipeline.stage_1_ingestion.IngestionPipeline.process_document",
            new_callable=AsyncMock,
            return_value={"status": "success", "doc_id": "test_upload", "chunks": 1},
        ):
            with open(sample_pdf, "rb") as pdf_f, open(txt_file, "rb") as txt_f:
                resp = test_client.post(
                    "/api/v3/ingest_pdf/",
                    files=[
                        ("files", ("test_upload.pdf", pdf_f, "application/pdf")),
                        ("files", ("readme.txt", txt_f, "text/plain")),
                    ],
                )
        body = resp.json()
        statuses = [r["status"] for r in body["job_summary"]]
        assert "success" in statuses
        assert "skipped" in statuses


# =====================================================================
#  Audio ingestion
# =====================================================================

class TestIngestAudio:
    """Upload a single audio file for transcription."""

    def test_valid_wav_upload(self, test_client, tmp_path):
        """A .wav file should be accepted."""
        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"\x00" * 100)

        with open(wav, "rb") as f:
            resp = test_client.post(
                "/api/v3/ingest_audio/",
                files=[("file", ("clip.wav", f, "audio/wav"))],
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_valid_mp3_upload(self, test_client, tmp_path):
        """A .mp3 file should be accepted."""
        mp3 = tmp_path / "track.mp3"
        mp3.write_bytes(b"\x00" * 80)

        with open(mp3, "rb") as f:
            resp = test_client.post(
                "/api/v3/ingest_audio/",
                files=[("file", ("track.mp3", f, "audio/mpeg"))],
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_unsupported_audio_format(self, test_client, tmp_path):
        """A .ogg file (not in the allow-list) should be skipped."""
        ogg = tmp_path / "clip.ogg"
        ogg.write_bytes(b"\x00" * 50)

        with open(ogg, "rb") as f:
            resp = test_client.post(
                "/api/v3/ingest_audio/",
                files=[("file", ("clip.ogg", f, "audio/ogg"))],
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"
