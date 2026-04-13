import os
import io
import uuid
import json
import base64
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional
from github import Github
import firebase_admin
from firebase_admin import credentials, firestore

# Setup Gemini API key
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Pydantic schemas for Gemini structured output
class QuestionExtraction(BaseModel):
    question_text: str = Field(description="The exact text of the question.")
    marks: str = Field(description="The marks assigned to the question, e.g. '2', '3', '5', etc. If not explicitly found, try to infer or leave empty.")
    page_number: int = Field(description="The page number where the question was found (0-indexed based on the provided image).")
    bounding_box: List[float] = Field(description="The bounding box coordinates [xmin, ymin, xmax, ymax] normalized between 0.0 and 1.0 (where 0,0 is top-left).")

class ChapterQuestionsList(BaseModel):
    questions: List[QuestionExtraction] = Field(description="List of questions found for the target chapter.")

class VisualAgent:
    def __init__(self, github_token: str = None, repo_name: str = None, firestore_creds: dict = None):
        """
        Initialize the Visual Agent.
        """
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.repo_name = repo_name or os.environ.get("GITHUB_REPO")
        if self.github_token:
            self.github = Github(self.github_token)
            self.repo = self.github.get_repo(self.repo_name) if self.repo_name else None
        else:
            self.github = None
            self.repo = None

        self.db = None
        if firestore_creds:
            if not firebase_admin._apps:
                cred = credentials.Certificate(firestore_creds)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()

    def _fetch_pdf(self, source: str) -> bytes:
        """Fetch PDF either from a URL or read from local path."""
        if source.startswith('http://') or source.startswith('https://'):
            response = requests.get(source)
            response.raise_for_status()
            return response.content
        else:
            with open(source, 'rb') as f:
                return f.read()

    def process_chapter_pdfs(self, pdf_sources: List[str], target_chapter: str, chapter_id: str, grade: str, subject: str, year: str) -> dict:
        """
        Main pipeline: Ingest, Process, AI Vision, Automated Snipping, Dual-Sync Storage.
        Returns a summary of operations.
        """
        summary = {
            "pdfs_processed": 0,
            "questions_extracted": 0,
            "successful_uploads": 0,
            "errors": []
        }

        for source in pdf_sources:
            try:
                pdf_bytes = self._fetch_pdf(source)
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:

                    # Iterate through pages
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        # Get image of the page
                        pix = page.get_pixmap(dpi=150) # Moderate DPI for Gemini to process quickly
                        img_data = pix.tobytes("jpeg")

                        # AI Vision step
                        extracted_questions = self._analyze_page_with_gemini(img_data, target_chapter)

                        if not extracted_questions or not extracted_questions.questions:
                            continue

                        for q in extracted_questions.questions:
                            # Automated Snipping
                            try:
                                # Map normalized coords to actual PDF coords
                                rect = self._get_rect_from_normalized(q.bounding_box, page.rect)

                                # Crop and save as high-res PNG
                                snip_pix = page.get_pixmap(clip=rect, dpi=300)
                                snip_bytes = snip_pix.tobytes("png")

                                question_id = str(uuid.uuid4())

                                # Update exact page number instead of generic (in case we want absolute vs relative)
                                q.page_number = page_num

                                # Dual-Sync Storage
                                github_url = self._upload_to_github(snip_bytes, grade, subject, target_chapter, question_id)

                                if github_url:
                                    self._save_to_firestore(q, chapter_id, year, github_url, question_id)
                                    summary["successful_uploads"] += 1

                                summary["questions_extracted"] += 1
                            except Exception as e:
                                summary["errors"].append(f"Error processing question on page {page_num}: {str(e)}")

                    summary["pdfs_processed"] += 1
            except Exception as e:
                summary["errors"].append(f"Error processing PDF {source}: {str(e)}")

        return summary

    def _analyze_page_with_gemini(self, img_data: bytes, target_chapter: str) -> Optional[ChapterQuestionsList]:
        """Use Gemini 3.1 Flash-Lite to extract questions."""
        # Using the standard SDK pattern for multimodal with structured output
        try:
            model = genai.GenerativeModel("gemini-3.1-flash-lite")

            prompt = f"""
            Analyze the provided exam page.
            Your task is to identify every question that belongs to the chapter: '{target_chapter}'.
            For each matching question, extract:
            1. The exact question text.
            2. The marks for the question.
            3. The bounding box coordinates of the question on the page, in the format [xmin, ymin, xmax, ymax], normalized between 0.0 and 1.0. 0.0,0.0 is the top left. The bounding box should cover the entire question text and any associated diagram.
            """

            # Using prompt + image
            response = model.generate_content(
                [
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_data}
                ],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ChapterQuestionsList,
                    temperature=0.1,
                ),
            )

            if response.text:
                data = json.loads(response.text)
                return ChapterQuestionsList(**data)
            return None
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    def _get_rect_from_normalized(self, bbox: List[float], page_rect: fitz.Rect) -> fitz.Rect:
        """Convert normalized [xmin, ymin, xmax, ymax] to fitz.Rect based on page dimensions."""
        if len(bbox) != 4:
            raise ValueError("Bounding box must have 4 coordinates.")

        xmin, ymin, xmax, ymax = bbox

        # Clamp values between 0.0 and 1.0 to prevent out of bounds
        xmin = max(0.0, min(1.0, xmin))
        ymin = max(0.0, min(1.0, ymin))
        xmax = max(0.0, min(1.0, xmax))
        ymax = max(0.0, min(1.0, ymax))

        # Calculate absolute coordinates
        x0 = xmin * page_rect.width + page_rect.x0
        y0 = ymin * page_rect.height + page_rect.y0
        x1 = xmax * page_rect.width + page_rect.x0
        y1 = ymax * page_rect.height + page_rect.y0

        return fitz.Rect(x0, y0, x1, y1)

    def _upload_to_github(self, image_bytes: bytes, grade: str, subject: str, chapter: str, question_id: str) -> str:
        """Upload the PNG to GitHub and return the raw URL."""
        if not self.repo:
            print("GitHub not configured, skipping upload.")
            return f"mock_url_for_{question_id}"

        try:
            # cdn/pyq_snips/{grade}/{subject}/{chapter}/{question_id}.png
            # Clean up chapter name for URL
            clean_chapter = chapter.replace(" ", "_").lower()
            path = f"cdn/pyq_snips/{grade}/{subject}/{clean_chapter}/{question_id}.png"

            self.repo.create_file(
                path=path,
                message=f"Add question snippet for {chapter}",
                content=image_bytes,
                branch="main"  # Or use default branch
            )

            # Construct raw GitHub content URL
            # Format: https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{path}
            # Or use jsDelivr, etc. We'll use standard raw.githubusercontent for now.
            repo_full_name = self.repo.full_name
            return f"https://raw.githubusercontent.com/{repo_full_name}/main/{path}"
        except Exception as e:
            print(f"GitHub upload failed: {e}")
            return ""

    def _save_to_firestore(self, question: QuestionExtraction, chapter_id: str, year: str, github_url: str, question_id: str):
        """Save question metadata to Firestore."""
        if not self.db:
            print("Firestore not configured, skipping DB write.")
            return

        try:
            doc_ref = self.db.collection("Chapter_Analysis").document(chapter_id).collection("Historical_Questions").document(question_id)
            doc_ref.set({
                "question_text": question.question_text,
                "marks": question.marks,
                "page_number": question.page_number,
                "year": year,
                "github_cdn_url": github_url,
                "coordinates": question.bounding_box,
            })
        except Exception as e:
            print(f"Firestore save failed: {e}")
