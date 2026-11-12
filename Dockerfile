# The scoring service's image.
#   build:  docker build -t credit-risk-api .
#   run:    docker run -p 8080:8080 -v "$PWD/models:/app/models" credit-risk-api
#
# PROVIDED: a slim base image. Use the SAME Python minor version as the laptop that exported
# the model (the second line of requirements-api.txt says which), so the saved model loads.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080
WORKDIR /app

# TODO(student): install the pinned requirements.
#   Copy requirements-api.txt into the image, then
#   RUN pip install --no-cache-dir -r requirements-api.txt
#   (Copying it BEFORE the code lets Docker reuse this slow step when only the code changes.)

# PROVIDED: only the service's code goes into the image. No data, no model (the service
# downloads it from your bucket when it starts), no tests, no notebooks.
COPY api/ api/

# TODO(student): create a user without root rights, and switch to it.
#   RUN useradd --create-home appuser
#   USER appuser
#   (If someone breaks into the service, they should not be root inside the container.)

# TODO(student): start the service: uvicorn serving api.main:app on host 0.0.0.0 and port
#   $PORT (Cloud Run tells the container which port to use through PORT).
#   CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
