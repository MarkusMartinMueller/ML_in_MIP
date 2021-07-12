call .\build.bat

docker volume create cada_detection-output

docker run --rm^
 --memory=4g^
 -v %~dp0\test\:/input/^
 -v cada_detection-output:/output/^
 cada_detection

docker run --rm^
 -v cada_detection-output:/output/^
 python:3.7-slim cat /output/metrics.json | python -m json.tool

docker volume rm cada_detection-output
