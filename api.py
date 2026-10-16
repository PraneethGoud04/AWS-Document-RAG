
import hashlib
import json
import os
import uuid

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from botocore.exceptions import ClientError, BotoCoreError

load_dotenv()

app = FastAPI(title="AWS Document RAG API")

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
ANSWERER_FUNCTION_NAME = os.getenv(
    "ANSWERER_FUNCTION_NAME",
    "document-rag-answerer"
)

s3 = boto3.client("s3", region_name=AWS_REGION)
lambda_client = boto3.client("lambda", region_name=AWS_REGION)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


def find_duplicate_pdf(file_hash: str):
    """Find an identical PDF already stored under uploads/."""
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=S3_BUCKET_NAME,
        Prefix="uploads/"
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.lower().endswith(".pdf"):
                continue

            head = s3.head_object(
                Bucket=S3_BUCKET_NAME,
                Key=key
            )

            stored_hash = head.get("Metadata", {}).get("sha256")

            if stored_hash:
                if stored_hash == file_hash:
                    return key
                continue

            # Older files may not have SHA-256 metadata.
            response = s3.get_object(
                Bucket=S3_BUCKET_NAME,
                Key=key
            )
            existing_content = response["Body"].read()
            existing_hash = hashlib.sha256(
                existing_content
            ).hexdigest()

            if existing_hash == file_hash:
                return key

    return None


def check_s3_object(key: str):
    """Return True if an S3 object exists, False if it is missing."""
    try:
        s3.head_object(Bucket=S3_BUCKET_NAME, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def get_document_status(document_id: str):
    """Check UUID-based pipeline output."""
    prefix = f"extracted/{document_id}/"

    if check_s3_object(f"{prefix}embeddings.json"):
        return {
            "document_id": document_id,
            "status": "ready",
            "message": "Document embeddings are available."
        }

    if check_s3_object(f"{prefix}chunks.json"):
        return {
            "document_id": document_id,
            "status": "embedding_in_progress",
            "message": "Text extraction completed; waiting for embeddings."
        }

    if check_s3_object(f"{prefix}pages.json"):
        return {
            "document_id": document_id,
            "status": "processing",
            "message": "Page text exists; waiting for chunking or embeddings."
        }

    return None


@app.get("/")
def home():
    return {"message": "AWS Document RAG API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    if not S3_BUCKET_NAME:
        raise HTTPException(
            status_code=500,
            detail="S3_BUCKET_NAME is not configured."
        )

    try:
        content = await file.read()

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file does not appear to be a valid PDF."
            )

        file_hash = hashlib.sha256(content).hexdigest()

        try:
            duplicate_key = find_duplicate_pdf(file_hash)
        except (ClientError, BotoCoreError) as exc:
            print("Duplicate check error:", str(exc))
            raise HTTPException(
                status_code=502,
                detail="Could not check existing PDFs for duplicates."
            )

        if duplicate_key:
            existing_document_id = os.path.basename(
                duplicate_key
            ).removesuffix(".pdf")

            return {
                "message": "This PDF has already been uploaded.",
                "duplicate": True,
                "existing_document_id": existing_document_id,
                "existing_s3_key": duplicate_key,
                "filename": file.filename,
                "processing_status": "already_uploaded"
            }

        document_id = str(uuid.uuid4())
        s3_key = f"uploads/{document_id}.pdf"

        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=content,
            ContentType="application/pdf",
            Metadata={
                "original-filename": file.filename[:200],
                "sha256": file_hash
            }
        )

        return {
            "message": "PDF uploaded successfully",
            "duplicate": False,
            "document_id": document_id,
            "filename": file.filename,
            "s3_bucket": S3_BUCKET_NAME,
            "s3_key": s3_key,
            "processing_status": "uploaded"
        }

    except HTTPException:
        raise
    except (ClientError, BotoCoreError) as exc:
        print("S3 upload error:", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Could not upload or check the PDF in S3."
        )
    finally:
        await file.close()


@app.get("/status/{document_id}")
def document_status(document_id: str):
    if not S3_BUCKET_NAME:
        raise HTTPException(
            status_code=500,
            detail="S3_BUCKET_NAME is not configured."
        )

    # UUID uploads use extracted/<uuid>/.
    try:
        uuid.UUID(document_id)
        is_uuid = True
    except (ValueError, TypeError, AttributeError):
        is_uuid = False

    try:
        if is_uuid:
            result = get_document_status(document_id)
            if result:
                return result

        # Legacy filename-based output folders.
        # Only allow a simple filename stem, not an arbitrary S3 path.
        safe_stem = (
            not is_uuid
            and document_id
            and len(document_id) <= 200
            and all(c.isalnum() or c in "_-" for c in document_id)
        )

        if safe_stem:
            prefix = f"extracted/{document_id}/"

            if check_s3_object(f"{prefix}embeddings.json"):
                return {
                    "document_id": document_id,
                    "status": "ready",
                    "message": "Document embeddings are available."
                }

            if check_s3_object(f"{prefix}chunks.json"):
                return {
                    "document_id": document_id,
                    "status": "embedding_in_progress",
                    "message": "Text extraction completed; waiting for embeddings."
                }

            if check_s3_object(f"{prefix}pages.json"):
                return {
                    "document_id": document_id,
                    "status": "processing",
                    "message": "Page text exists; waiting for chunking or embeddings."
                }

        return {
            "document_id": document_id,
            "status": "processing",
            "message": (
                "No processing output was found for this ID. "
                "Verify the S3 output folder and processor logs."
            )
        }

    except (ClientError, BotoCoreError) as exc:
        print("Status check error:", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Could not check document processing status."
        )


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please provide a question."
        )

    try:
        result = lambda_client.invoke(
            FunctionName=ANSWERER_FUNCTION_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps({
                "question": question,
                "top_k": request.top_k
            }).encode("utf-8")
        )

        if result.get("FunctionError"):
            error_text = result["Payload"].read().decode("utf-8")
            print("Answerer Lambda error:", error_text)
            raise HTTPException(
                status_code=502,
                detail="The answer-generation service failed."
            )

        lambda_result = json.loads(result["Payload"].read())
        status_code = lambda_result.get("statusCode", 500)
        body = lambda_result.get("body", {})

        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = {"message": body}

        if status_code != 200:
            error_message = (
                body.get("error", "Could not generate an answer.")
                if isinstance(body, dict)
                else "Could not generate an answer."
            )
            raise HTTPException(
                status_code=502,
                detail=error_message
            )

        return body

    except HTTPException:
        raise
    except (ClientError, BotoCoreError) as exc:
        print("Answerer invocation error:", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the answer-generation service."
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print("Invalid answerer response:", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Received an invalid response from the answer service."
        )
