# Similarity Search Engine
Similarity calculation for given two images with deep learning algorithm.
Using cosine metric as similarity degree.
Deep learning model is served with FastAPI for production.

## Usage
After installation of required packages run below command:

```bash
uvicorn app.main:app --reload
```

You need to convert your image file(png, jpeg or jpg etc.) to base64 format.
Then you can go to http://127.0.0.1:8000/docs adress for sending requests.

Sample request body for /predict request:
```
{
  "query": {"first_image": "/9j/4AAQSkZJRgABAQAA..."
            "second_image": "/9j/2wBDAAMCAgMCAgM..."
            }
}
```

## Acknowledgment
Part of REST API implementation is modified from [this tutorial](https://www.youtube.com/watch?v=56qQNcHJxyQ).