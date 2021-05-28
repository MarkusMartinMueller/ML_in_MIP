# set base image (host OS)
FROM python:3.6.9

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
RUN pip install torch-geometric

# copy the content of the local src directory to the working directory
#COPY src/ .

ENV PYTHONPATH ="/project"

# command to run on container start
CMD [ "python"]